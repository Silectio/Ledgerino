import streamlit as st
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import pandas as pd
import math
import json
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
import psycopg2
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page
st.set_page_config(
    page_title="Ledger minimal",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Utilitaires
def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# Configuration de la base de données
@st.cache_resource
def get_database_connection():
    """Initialise la connexion à la base de données Neon."""
    try:
        # Récupère l'URL de la base de données depuis les secrets
        database_url = st.secrets["DATABASE_URL"]
        engine = create_engine(database_url, echo=False)

        # Teste la connexion
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        return engine
    except Exception as e:
        st.error(f"Erreur de connexion à la base de données: {e}")
        st.stop()


def init_database_tables():
    """Initialise les tables de la base de données."""
    engine = get_database_connection()

    # Vérifie si les tables existent déjà
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    if not all(
        table in existing_tables
        for table in ["users", "accounts", "ledger_entries", "rules"]
    ):
        with engine.begin() as conn:
            # Table des utilisateurs
            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS users (
                    pseudo VARCHAR(50) PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
                )
            )

            # Table des comptes
            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS accounts (
                    id SERIAL PRIMARY KEY,
                    pseudo VARCHAR(50) REFERENCES users(pseudo) ON DELETE CASCADE,
                    account_id VARCHAR(50) NOT NULL,
                    name VARCHAR(100) NOT NULL,
                    is_unlinked BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(pseudo, account_id)
                )
            """
                )
            )

            # Table du ledger
            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS ledger_entries (
                    id SERIAL PRIMARY KEY,
                    pseudo VARCHAR(50) REFERENCES users(pseudo) ON DELETE CASCADE,
                    ts TIMESTAMP NOT NULL,
                    type VARCHAR(20) NOT NULL,
                    amount_cents INTEGER NOT NULL,
                    src_account_id VARCHAR(50),
                    dest_account_id VARCHAR(50),
                    account_id VARCHAR(50),
                    note TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
                )
            )

            # Table des règles
            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS rules (
                    id SERIAL PRIMARY KEY,
                    pseudo VARCHAR(50) REFERENCES users(pseudo) ON DELETE CASCADE,
                    rule_id VARCHAR(50) NOT NULL,
                    name VARCHAR(100) NOT NULL,
                    require_value BOOLEAN DEFAULT FALSE,
                    default_amount_cents INTEGER DEFAULT 0,
                    trigger_label VARCHAR(100) DEFAULT 'Exécuter',
                    use_balance_difference BOOLEAN DEFAULT FALSE,
                    actions JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(pseudo, rule_id)
                )
            """
                )
            )


def migrate_database():
    """Applique les migrations de base de données nécessaires."""
    engine = get_database_connection()
    with engine.begin() as conn:
        # Migration 1 : Ajouter la colonne use_balance_difference aux règles si elle n'existe pas
        try:
            result = conn.execute(
                text(
                    """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'rules' AND column_name = 'use_balance_difference'
            """
                )
            )

            if not result.fetchone():
                conn.execute(
                    text(
                        """
                    ALTER TABLE rules ADD COLUMN use_balance_difference BOOLEAN DEFAULT FALSE
                """
                    )
                )
                print(
                    "✅ Migration appliquée : ajout de la colonne use_balance_difference aux règles"
                )
            else:
                print(
                    "ℹ️ Migration déjà appliquée : colonne use_balance_difference existe dans rules"
                )

        except Exception as e:
            print(f"⚠️ Erreur de migration rules.use_balance_difference : {e}")

        # Migration 2 : Ajouter la colonne is_unlinked aux comptes si elle n'existe pas
        try:
            result = conn.execute(
                text(
                    """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'accounts' AND column_name = 'is_unlinked'
            """
                )
            )

            if not result.fetchone():
                conn.execute(
                    text(
                        """
                    ALTER TABLE accounts ADD COLUMN is_unlinked BOOLEAN DEFAULT FALSE
                """
                    )
                )
                print(
                    "✅ Migration appliquée : ajout de la colonne is_unlinked aux comptes"
                )
            else:
                print(
                    "ℹ️ Migration déjà appliquée : colonne is_unlinked existe dans accounts"
                )

        except Exception as e:
            print(f"⚠️ Erreur de migration accounts.is_unlinked : {e}")


# Initialise la base de données au démarrage
init_database_tables()

# Applique les migrations nécessaires
migrate_database()


# Fonctions de base de données
def create_user_if_not_exists(pseudo: str):
    """Crée un utilisateur s'il n'existe pas déjà."""
    engine = get_database_connection()
    with engine.begin() as conn:
        # Vérifier si l'utilisateur existe
        result = conn.execute(
            text("SELECT pseudo FROM users WHERE pseudo = :pseudo"), {"pseudo": pseudo}
        )
        if not result.fetchone():
            # Créer l'utilisateur
            conn.execute(
                text("INSERT INTO users (pseudo) VALUES (:pseudo)"), {"pseudo": pseudo}
            )
            # Créer le compte par défaut
            conn.execute(
                text(
                    """
                INSERT INTO accounts (pseudo, account_id, name) 
                VALUES (:pseudo, 'general', 'Général')
            """
                ),
                {"pseudo": pseudo},
            )


def load_user_accounts(pseudo: str) -> List[Dict]:
    """Charge les comptes d'un utilisateur."""
    engine = get_database_connection()
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
            SELECT account_id, name, is_unlinked FROM accounts 
            WHERE pseudo = :pseudo 
            ORDER BY created_at
        """
            ),
            {"pseudo": pseudo},
        )
        return [
            {
                "account_id": row.account_id,
                "name": row.name,
                "is_unlinked": row.is_unlinked or False,
            }
            for row in result
        ]


def load_user_ledger(pseudo: str) -> List[Dict]:
    """Charge le ledger d'un utilisateur."""
    engine = get_database_connection()
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
            SELECT ts, type, amount_cents, src_account_id, dest_account_id, account_id, note
            FROM ledger_entries 
            WHERE pseudo = :pseudo 
            ORDER BY ts, id
        """
            ),
            {"pseudo": pseudo},
        )

        ledger_entries = []
        for row in result:
            entry = dict(row._mapping)
            # S'assurer que ts est une chaîne ISO si c'est un datetime
            if isinstance(entry.get("ts"), datetime):
                entry["ts"] = entry["ts"].isoformat()
            ledger_entries.append(entry)

        return ledger_entries


def load_user_rules(pseudo: str) -> List[Dict]:
    """Charge les règles d'un utilisateur."""
    engine = get_database_connection()
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
            SELECT rule_id, name, require_value, default_amount_cents, trigger_label, use_balance_difference, actions
            FROM rules 
            WHERE pseudo = :pseudo 
            ORDER BY created_at
        """
            ),
            {"pseudo": pseudo},
        )
        rules = []
        for row in result:
            rule = dict(row._mapping)
            # Convertir le JSON des actions
            if isinstance(rule["actions"], str):
                rule["actions"] = json.loads(rule["actions"])
            rules.append(rule)
        return rules


def save_account(pseudo: str, account_id: str, name: str, is_unlinked: bool = False):
    """Sauvegarde un nouveau compte."""
    engine = get_database_connection()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
            INSERT INTO accounts (pseudo, account_id, name, is_unlinked) 
            VALUES (:pseudo, :account_id, :name, :is_unlinked)
        """
            ),
            {
                "pseudo": pseudo,
                "account_id": account_id,
                "name": name,
                "is_unlinked": is_unlinked,
            },
        )


def delete_account(pseudo: str, account_id: str):
    """Supprime un compte."""
    engine = get_database_connection()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
            DELETE FROM accounts 
            WHERE pseudo = :pseudo AND account_id = :account_id
        """
            ),
            {"pseudo": pseudo, "account_id": account_id},
        )


def save_ledger_entry(pseudo: str, entry: Dict):
    """Sauvegarde une entrée dans le ledger."""
    engine = get_database_connection()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
            INSERT INTO ledger_entries 
            (pseudo, ts, type, amount_cents, src_account_id, dest_account_id, account_id, note)
            VALUES (:pseudo, :ts, :type, :amount_cents, :src_account_id, :dest_account_id, :account_id, :note)
        """
            ),
            {
                "pseudo": pseudo,
                "ts": entry.get("ts"),
                "type": entry.get("type"),
                "amount_cents": entry.get("amount_cents"),
                "src_account_id": entry.get("src_account_id"),
                "dest_account_id": entry.get("dest_account_id"),
                "account_id": entry.get("account_id"),
                "note": entry.get("note"),
            },
        )


def delete_last_ledger_entry(pseudo: str) -> bool:
    """Supprime la dernière entrée du ledger."""
    engine = get_database_connection()
    with engine.begin() as conn:
        # Obtenir l'ID de la dernière entrée
        result = conn.execute(
            text(
                """
            SELECT id FROM ledger_entries 
            WHERE pseudo = :pseudo 
            ORDER BY ts DESC, id DESC 
            LIMIT 1
        """
            ),
            {"pseudo": pseudo},
        )
        last_entry = result.fetchone()

        if last_entry:
            conn.execute(
                text(
                    """
                DELETE FROM ledger_entries 
                WHERE id = :entry_id
            """
                ),
                {"entry_id": last_entry.id},
            )
            return True
        return False


def save_rule(pseudo: str, rule: Dict):
    """Sauvegarde une nouvelle règle."""
    engine = get_database_connection()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
            INSERT INTO rules 
            (pseudo, rule_id, name, require_value, default_amount_cents, trigger_label, use_balance_difference, actions)
            VALUES (:pseudo, :rule_id, :name, :require_value, :default_amount_cents, :trigger_label, :use_balance_difference, :actions)
        """
            ),
            {
                "pseudo": pseudo,
                "rule_id": rule.get("rule_id"),
                "name": rule.get("name"),
                "require_value": rule.get("require_value", False),
                "default_amount_cents": rule.get("default_amount_cents", 0),
                "trigger_label": rule.get("trigger_label", "Exécuter"),
                "use_balance_difference": rule.get("use_balance_difference", False),
                "actions": json.dumps(rule.get("actions", [])),
            },
        )


def update_rule(pseudo: str, rule: Dict):
    """Met à jour une règle existante."""
    engine = get_database_connection()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
            UPDATE rules 
            SET name = :name, 
                require_value = :require_value, 
                default_amount_cents = :default_amount_cents, 
                trigger_label = :trigger_label, 
                use_balance_difference = :use_balance_difference,
                actions = :actions
            WHERE pseudo = :pseudo AND rule_id = :rule_id
        """
            ),
            {
                "pseudo": pseudo,
                "rule_id": rule.get("rule_id"),
                "name": rule.get("name"),
                "require_value": rule.get("require_value", False),
                "default_amount_cents": rule.get("default_amount_cents", 0),
                "trigger_label": rule.get("trigger_label", "Exécuter"),
                "use_balance_difference": rule.get("use_balance_difference", False),
                "actions": json.dumps(rule.get("actions", [])),
            },
        )


def delete_rule(pseudo: str, rule_id: str):
    """Supprime une règle."""
    engine = get_database_connection()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
            DELETE FROM rules 
            WHERE pseudo = :pseudo AND rule_id = :rule_id
        """
            ),
            {"pseudo": pseudo, "rule_id": rule_id},
        )


# Fonctions d'analyse pour le dashboard
def compute_balances_at_date(
    ledger_entries: List[Dict],
    accounts: List[Dict],
    target_date: Optional[datetime] = None,
) -> Dict[str, int]:
    """Calcule les soldes des comptes à une date donnée (ou maintenant si None)."""
    balances: Dict[str, int] = {acc["account_id"]: 0 for acc in accounts}

    for op in ledger_entries:
        # Parser la date de l'opération
        ts_value = op.get("ts")
        if not ts_value:
            continue  # Ignorer les entrées sans timestamp

        # Gérer les deux cas : datetime object ou string
        if isinstance(ts_value, datetime):
            op_date = ts_value
        elif isinstance(ts_value, str):
            if not ts_value.strip():
                continue
            try:
                op_date = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
            except ValueError:
                # Si le format de date est invalide, ignorer cette entrée
                continue
        else:
            continue

        # Normaliser les dates pour la comparaison (s'assurer qu'elles sont toutes timezone-aware)
        if target_date:
            # Normaliser op_date
            if op_date.tzinfo is None:
                op_date = op_date.replace(tzinfo=timezone.utc)

            # Créer une version normalisée de target_date sans la modifier
            target_date_normalized = target_date
            if target_date_normalized.tzinfo is None:
                target_date_normalized = target_date_normalized.replace(
                    tzinfo=timezone.utc
                )

            if op_date > target_date_normalized:
                continue

        t = op.get("type")
        amt = int(op.get("amount_cents", 0))

        if t == "deposit":
            acc_id = op.get("dest_account_id")
            if acc_id in balances:
                balances[acc_id] += amt
        elif t == "expense":
            acc_id = op.get("src_account_id")
            if acc_id in balances:
                balances[acc_id] -= amt
        elif t == "transfer":
            src_id = op.get("src_account_id")
            dest_id = op.get("dest_account_id")
            if src_id in balances:
                balances[src_id] -= amt
            if dest_id in balances:
                balances[dest_id] += amt
        elif t == "adjustment":
            acc_id = op.get("account_id")
            if acc_id in balances:
                balances[acc_id] += amt

    return balances


def get_balance_evolution(
    ledger_entries: List[Dict], accounts: List[Dict], days: int = 30
) -> pd.DataFrame:
    """Récupère l'évolution du solde total sur les N derniers jours."""
    if not ledger_entries:
        return pd.DataFrame()

    # Générer les dates
    today = datetime.now(timezone.utc)
    dates = [today - timedelta(days=i) for i in range(days, -1, -1)]

    evolution_data = []

    for date in dates:
        balances = compute_balances_at_date(ledger_entries, accounts, date)
        total_cents = sum(balances.values())

        evolution_data.append(
            {
                "date": date.date(),
                "total_euros": total_cents / 100.0,
                "total_cents": total_cents,
            }
        )

    return pd.DataFrame(evolution_data)


def get_daily_flow_analysis(ledger_entries: List[Dict], days: int = 30) -> pd.DataFrame:
    """Analyse les flux quotidiens (dépenses et recettes)."""
    if not ledger_entries:
        return pd.DataFrame()

    # Filtrer les derniers jours
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    recent_entries = []

    for op in ledger_entries:
        ts_value = op.get("ts")
        if not ts_value:
            continue

        # Gérer les deux cas : datetime object ou string
        if isinstance(ts_value, datetime):
            op_date = ts_value
        elif isinstance(ts_value, str):
            if not ts_value.strip():
                continue
            try:
                op_date = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
            except ValueError:
                continue
        else:
            continue

        # Normaliser les dates pour la comparaison (s'assurer qu'elles sont toutes timezone-aware)
        if op_date.tzinfo is None:
            op_date = op_date.replace(tzinfo=timezone.utc)

        if op_date >= cutoff_date:
            recent_entries.append(op)

    # Grouper par jour
    daily_flows = {}

    for op in recent_entries:
        ts_value = op.get("ts")
        if not ts_value:
            continue

        # Gérer les deux cas : datetime object ou string
        if isinstance(ts_value, datetime):
            op_date = ts_value.date()
        elif isinstance(ts_value, str):
            if not ts_value.strip():
                continue
            try:
                op_date = datetime.fromisoformat(ts_value.replace("Z", "+00:00")).date()
            except ValueError:
                continue
        else:
            continue

        if op_date not in daily_flows:
            daily_flows[op_date] = {"income": 0, "expense": 0, "net": 0}

        amt = int(op.get("amount_cents", 0))
        t = op.get("type")

        if t == "deposit":
            daily_flows[op_date]["income"] += amt
        elif t == "expense":
            daily_flows[op_date]["expense"] += amt
        # Pour les transfers et adjustments, on peut les ignorer ou les traiter différemment

    # Convertir en DataFrame
    flow_data = []
    for date, flows in daily_flows.items():
        flows["net"] = flows["income"] - flows["expense"]
        flow_data.append(
            {
                "date": date,
                "recettes": flows["income"] / 100.0,
                "depenses": flows["expense"] / 100.0,
                "net": flows["net"] / 100.0,
            }
        )

    return pd.DataFrame(flow_data).sort_values("date")


def calculate_kpis(ledger_entries: List[Dict], accounts: List[Dict]) -> Dict:
    """Calcule divers KPIs."""
    if not ledger_entries:
        return {}

    # Solde actuel
    current_balance = (
        sum(compute_balances_at_date(ledger_entries, accounts).values()) / 100.0
    )

    # Evolution sur 30 jours
    balance_30d = get_balance_evolution(ledger_entries, accounts, 30)

    # Moyenne mobile 7 jours
    if len(balance_30d) >= 7:
        avg_7d = balance_30d.tail(7)["total_euros"].mean()
    else:
        avg_7d = current_balance

    # Moyenne mobile 30 jours
    if len(balance_30d) >= 30:
        avg_30d = balance_30d["total_euros"].mean()
    else:
        avg_30d = current_balance

    # Tendance (différence entre aujourd'hui et il y a 7 jours)
    if len(balance_30d) >= 8:
        trend_7d = (
            balance_30d.iloc[-1]["total_euros"] - balance_30d.iloc[-8]["total_euros"]
        )
    else:
        trend_7d = 0

    # Analyse des flux des 30 derniers jours
    flows_30d = get_daily_flow_analysis(ledger_entries, 30)

    if not flows_30d.empty:
        total_income_30d = flows_30d["recettes"].sum()
        total_expense_30d = flows_30d["depenses"].sum()
        avg_daily_expense = flows_30d["depenses"].mean()
    else:
        total_income_30d = total_expense_30d = avg_daily_expense = 0

    return {
        "solde_actuel": current_balance,
        "moyenne_7j": avg_7d,
        "moyenne_30j": avg_30d,
        "tendance_7j": trend_7d,
        "recettes_30j": total_income_30d,
        "depenses_30j": total_expense_30d,
        "depense_moy_jour": avg_daily_expense,
    }


def predict_balance_evolution(
    balance_evolution: pd.DataFrame, prediction_days: int = 30
) -> pd.DataFrame:
    """Prédit l'évolution du solde pour les prochains jours basé sur la tendance actuelle."""
    if balance_evolution.empty or len(balance_evolution) < 2:
        return pd.DataFrame()

    # S'assurer que la colonne date est au bon format
    balance_evolution = balance_evolution.copy()
    balance_evolution["date"] = pd.to_datetime(balance_evolution["date"])

    # Calcul de la tendance linéaire
    balance_evolution["days_from_start"] = (
        balance_evolution["date"] - balance_evolution["date"].min()
    ).dt.days

    x = balance_evolution["days_from_start"]
    y = balance_evolution["total_euros"]

    if len(x) < 2:
        return pd.DataFrame()

    # Régression linéaire simple
    n = len(x)
    slope = ((x * y).sum() - x.sum() * y.sum() / n) / ((x * x).sum() - x.sum() ** 2 / n)
    intercept = y.mean() - slope * x.mean()

    # Générer les prédictions
    last_date = balance_evolution["date"].max()
    last_day_num = balance_evolution["days_from_start"].max()

    prediction_dates = [
        last_date + timedelta(days=i + 1) for i in range(prediction_days)
    ]
    prediction_days_num = [last_day_num + i + 1 for i in range(prediction_days)]
    prediction_values = [slope * day + intercept for day in prediction_days_num]

    predictions = pd.DataFrame(
        {
            "date": prediction_dates,
            "total_euros": prediction_values,
            "is_prediction": True,
        }
    )

    # Ajouter la colonne is_prediction aux données historiques
    balance_evolution["is_prediction"] = False

    # Combiner les données historiques et les prédictions
    return pd.concat(
        [balance_evolution[["date", "total_euros", "is_prediction"]], predictions],
        ignore_index=True,
    )


def get_balance_evolution_by_account(
    ledger_entries: List[Dict],
    accounts: Dict,
    days: int = 30,
    selected_accounts: List[str] = [],
) -> pd.DataFrame:
    """Obtient l'évolution du solde par compte sur une période donnée."""
    if not ledger_entries:
        return pd.DataFrame()

    # Filtrer les comptes si spécifié
    if selected_accounts:
        filtered_accounts = {
            k: v for k, v in accounts.items() if k in selected_accounts
        }
    else:
        filtered_accounts = accounts

    # Générer les dates
    today = datetime.now(timezone.utc)
    dates = [today - timedelta(days=i) for i in range(days, -1, -1)]

    evolution_data = []

    for date in dates:
        balances = compute_balances_at_date(
            ledger_entries, list(filtered_accounts.values()), date
        )

        for account_id, balance_cents in balances.items():
            if account_id in filtered_accounts:
                evolution_data.append(
                    {
                        "date": date.date(),
                        "account_id": account_id,
                        "account_name": filtered_accounts[account_id].get(
                            "name", account_id
                        ),
                        "balance_euros": balance_cents / 100.0,
                    }
                )

    return pd.DataFrame(evolution_data)


# État initial
if "accounts" not in st.session_state:
    st.session_state.accounts = []

if "ledger" not in st.session_state:
    st.session_state.ledger = []
# Fonctions de calcul d'intérêts pour le dashboard
def calculate_interest_simulation(
    balance_eur: float,
    annual_rate: float,
    period_months: int = 12,
    compound_frequency: int = 365,  # Capitalisation quotidienne
    nexo_percentage: float = 0.0  # % de NEXO tokens pour bonus
) -> Dict:
    """
    Calcule les intérêts composés avec possibilité de bonus NEXO.
    
    Args:
        balance_eur: Solde de départ en euros
        annual_rate: Taux annuel en % (ex: 8.5 pour 8.5%)
        period_months: Période de simulation en mois
        compound_frequency: Fréquence de capitalisation par an (365 = quotidien)
        nexo_percentage: % du portefeuille en tokens NEXO (0-100%)
    
    Returns:
        Dict contenant les résultats de simulation
    """
    if balance_eur <= 0:
        return {
            "initial_balance": 0,
            "final_balance": 0,
            "total_interest": 0,
            "monthly_interest": 0,
            "effective_rate": annual_rate,
            "nexo_bonus": 0
        }
    
    # Calculer le bonus NEXO (jusqu'à +2% selon le % de NEXO détenu)
    nexo_bonus = 0
    if nexo_percentage >= 10:  # Seuil minimum pour bonus
        # Bonus progressif: 0.5% pour 10% NEXO, jusqu'à 2% pour 20%+ NEXO
        nexo_bonus = min(2.0, (nexo_percentage / 10) * 0.5)
    
    effective_rate = annual_rate + nexo_bonus
    
    # Formule des intérêts composés: A = P(1 + r/n)^(nt)
    # P = capital initial, r = taux annuel, n = fréquence, t = temps en années
    rate_decimal = effective_rate / 100
    time_years = period_months / 12
    
    final_balance = balance_eur * (1 + rate_decimal / compound_frequency) ** (compound_frequency * time_years)
    total_interest = final_balance - balance_eur
    monthly_interest = total_interest / period_months if period_months > 0 else 0
    
    return {
        "initial_balance": balance_eur,
        "final_balance": final_balance,
        "total_interest": total_interest,
        "monthly_interest": monthly_interest,
        "effective_rate": effective_rate,
        "nexo_bonus": nexo_bonus,
        "annual_rate": annual_rate
    }

def calculate_platform_interests(balances_by_type: Dict) -> Dict:
    """
    Calcule les simulations d'intérêts pour les différentes plateformes.
    
    Args:
        balances_by_type: Dict avec les soldes par type (linked/unlinked)
    
    Returns:
        Dict avec les simulations par plateforme
    """
    linked_balance = sum(balances_by_type["linked"].values()) / 100  # Conversion en euros
    unlinked_balance = sum(balances_by_type["unlinked"].values()) / 100
    
    # Configuration des plateformes avec leurs taux
    platforms = {
        "nexo": {
            "name": "Nexo (Linked)",
            "balance": linked_balance,
            "base_rate": 8.5,  # Taux de base Nexo
            "supports_nexo_bonus": True,
            "description": "Comptes opérationnels sur Nexo"
        },
        "numerai": {
            "name": "Numerai (Unlinked)",
            "balance": unlinked_balance * 0.3,  # Estimation 30% sur Numerai
            "base_rate": 12.0,  # Taux Numerai
            "supports_nexo_bonus": False,
            "description": "Staking NMR sur Numerai"
        },
        "bricks": {
            "name": "Bricks (Unlinked)",
            "balance": unlinked_balance * 0.4,  # Estimation 40% sur Bricks
            "base_rate": 6.5,  # Taux Bricks immobilier
            "supports_nexo_bonus": False,
            "description": "Investissement immobilier tokenisé"
        },
        "autres": {
            "name": "Autres Plateformes",
            "balance": unlinked_balance * 0.3,  # 30% restant
            "base_rate": 5.0,  # Taux conservateur
            "supports_nexo_bonus": False,
            "description": "Diversification autres plateformes"
        }
    }
    
    results = {}
    
    for platform_id, config in platforms.items():
        if config["balance"] > 0:
            # Simulations sur 1, 6 et 12 mois
            simulations = {}
            for months in [1, 6, 12]:
                if config["supports_nexo_bonus"]:
                    # Pour Nexo, simuler différents niveaux de NEXO tokens
                    nexo_levels = [0, 10, 15, 20]  # % de NEXO tokens
                    nexo_sims = {}
                    for nexo_pct in nexo_levels:
                        sim = calculate_interest_simulation(
                            config["balance"], 
                            config["base_rate"], 
                            months,
                            nexo_percentage=nexo_pct
                        )
                        nexo_sims[f"nexo_{nexo_pct}pct"] = sim
                    simulations[f"{months}m"] = nexo_sims
                else:
                    # Pour les autres plateformes, simulation simple
                    sim = calculate_interest_simulation(
                        config["balance"], 
                        config["base_rate"], 
                        months
                    )
                    simulations[f"{months}m"] = sim
            
            results[platform_id] = {
                "config": config,
                "simulations": simulations
            }
    
    return results

def calculate_custom_simulation(
    initial_capital: float,
    monthly_addition: float,
    annual_rate: float,
    period_months: int,
    nexo_percentage: float = 0.0,
    nexo_price: float = 1.0
) -> Dict:
    """
    Simule l'évolution d'un capital avec ajouts mensuels et intérêts composés.
    
    Args:
        initial_capital: Capital de départ en euros
        monthly_addition: Ajout mensuel en euros
        annual_rate: Taux annuel en %
        period_months: Période en mois
        nexo_percentage: % du portefeuille en NEXO tokens
        nexo_price: Prix du token NEXO en euros
    
    Returns:
        Dict contenant l'évolution mois par mois
    """
    # Calculer le bonus NEXO
    nexo_bonus = 0
    if nexo_percentage >= 10:
        nexo_bonus = min(2.0, (nexo_percentage / 10) * 0.5)
    
    effective_rate = annual_rate + nexo_bonus
    monthly_rate = effective_rate / 100 / 12  # Taux mensuel
    
    evolution = []
    current_capital = initial_capital
    total_added = 0
    total_interest = 0
    
    # Calculs NEXO
    nexo_tokens_needed = 0
    nexo_value_eur = 0
    
    for month in range(period_months + 1):
        # Calcul des tokens NEXO nécessaires pour maintenir le %
        if nexo_percentage > 0:
            nexo_value_eur = current_capital * (nexo_percentage / 100)
            nexo_tokens_needed = nexo_value_eur / nexo_price if nexo_price > 0 else 0
        
        # Ajouter l'évolution de ce mois
        evolution.append({
            "month": month,
            "capital": current_capital,
            "total_added": total_added,
            "total_interest": total_interest,
            "monthly_interest": current_capital * monthly_rate if month > 0 else 0,
            "nexo_tokens_needed": nexo_tokens_needed,
            "nexo_value_eur": nexo_value_eur,
            "effective_rate": effective_rate
        })
        
        # Calculs pour le mois suivant (sauf pour le dernier mois)
        if month < period_months:
            # Ajouter les intérêts mensuels
            monthly_interest = current_capital * monthly_rate
            total_interest += monthly_interest
            current_capital += monthly_interest
            
            # Ajouter l'apport mensuel
            current_capital += monthly_addition
            total_added += monthly_addition
    
    return {
        "evolution": evolution,
        "summary": {
            "initial_capital": initial_capital,
            "final_capital": current_capital,
            "total_added": total_added,
            "total_interest": total_interest,
            "effective_rate": effective_rate,
            "nexo_bonus": nexo_bonus,
            "final_nexo_tokens": nexo_tokens_needed,
            "final_nexo_value": nexo_value_eur
        }
    }

def calculate_nexo_requirements(
    linked_balance: float, 
    target_percentage: float, 
    nexo_price: float
) -> Dict:
    """
    Calcule les besoins en tokens NEXO pour atteindre un pourcentage cible.
    
    Args:
        linked_balance: Solde total des comptes linked en euros
        target_percentage: Pourcentage cible de NEXO (10, 15, 20...)
        nexo_price: Prix actuel du token NEXO en euros
    
    Returns:
        Dict avec les calculs NEXO
    """
    if nexo_price <= 0:
        return {"error": "Prix NEXO invalide"}
    
    target_nexo_value = linked_balance * (target_percentage / 100)
    tokens_needed = target_nexo_value / nexo_price
    
    # Calculer le bonus obtenu
    bonus = 0
    if target_percentage >= 10:
        bonus = min(2.0, (target_percentage / 10) * 0.5)
    
    return {
        "linked_balance": linked_balance,
        "target_percentage": target_percentage,
        "target_nexo_value_eur": target_nexo_value,
        "tokens_needed": tokens_needed,
        "nexo_price": nexo_price,
        "bonus_rate": bonus,
        "investment_cost": target_nexo_value
    }


if "rules" not in st.session_state:
    st.session_state.rules = []

if "new_rule_actions" not in st.session_state:
    st.session_state.new_rule_actions = []

# Variables pour l'édition des règles
if "editing_rule_id" not in st.session_state:
    st.session_state.editing_rule_id = None

if "edit_rule_actions" not in st.session_state:
    st.session_state.edit_rule_actions = []

# Pseudo utilisateur pour partitionner les données
if "pseudo" not in st.session_state:
    st.session_state.pseudo = None

# Choix de la vue
page = st.sidebar.radio("Vue", ["Ledger", "📊 Dashboard", "Règles"], index=0)

# Sélection/chargement utilisateur
st.sidebar.markdown("---")
st.sidebar.subheader("Utilisateur")
user_input = st.sidebar.text_input(
    "Pseudo", value=st.session_state.pseudo or "", placeholder="ex: coco"
)
if st.sidebar.button("Charger cet utilisateur"):
    pseudo = user_input.strip()
    if not pseudo:
        st.sidebar.error("Saisir un pseudo")
    else:
        try:
            # Créer l'utilisateur s'il n'existe pas
            create_user_if_not_exists(pseudo)

            # Charger les données utilisateur
            st.session_state.pseudo = pseudo
            st.session_state.accounts = load_user_accounts(pseudo)
            st.session_state.ledger = load_user_ledger(pseudo)
            st.session_state.rules = load_user_rules(pseudo)

            st.sidebar.success(f"Utilisateur '{pseudo}' chargé avec succès")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Erreur lors du chargement: {e}")


# Utilitaires
def account_index(account_id: str) -> Optional[int]:
    for i, acc in enumerate(st.session_state.accounts):
        if acc["account_id"] == account_id:
            return i
    return None


def compute_balances() -> Dict[str, int]:
    """Agrège le ledger pour retourner le solde courant de chaque compte (en cents)."""
    balances: Dict[str, int] = {
        acc["account_id"]: 0 for acc in st.session_state.accounts
    }
    for op in st.session_state.ledger:
        t = op.get("type")
        amt = int(op.get("amount_cents", 0))
        if t == "deposit":
            acc_id = op.get("dest_account_id")
            if acc_id in balances:
                balances[acc_id] += amt
        elif t == "expense":
            acc_id = op.get("src_account_id")
            if acc_id in balances:
                balances[acc_id] -= amt
        elif t == "transfer":
            src_id = op.get("src_account_id")
            dest_id = op.get("dest_account_id")
            if src_id in balances:
                balances[src_id] -= amt
            if dest_id in balances:
                balances[dest_id] += amt
        elif t == "adjustment":
            # amount_cents peut être signé
            acc_id = op.get("account_id")
            if acc_id in balances:
                balances[acc_id] += amt
    return balances


def get_linked_accounts() -> List[Dict]:
    """Retourne uniquement les comptes linked (non-unlinked)."""
    return [
        acc for acc in st.session_state.accounts if not acc.get("is_unlinked", False)
    ]


def get_unlinked_accounts() -> List[Dict]:
    """Retourne uniquement les comptes unlinked."""
    return [acc for acc in st.session_state.accounts if acc.get("is_unlinked", False)]


def compute_balances_by_type() -> Dict[str, Dict[str, int]]:
    """Calcule les soldes séparés en linked et unlinked."""
    balances = compute_balances()
    linked_balances = {}
    unlinked_balances = {}

    for acc in st.session_state.accounts:
        acc_id = acc["account_id"]
        balance = balances.get(acc_id, 0)

        if acc.get("is_unlinked", False):
            unlinked_balances[acc_id] = balance
        else:
            linked_balances[acc_id] = balance

    return {"linked": linked_balances, "unlinked": unlinked_balances}



if not st.session_state.pseudo:
    st.warning(
        "Saisissez un pseudo dans la barre latérale puis cliquez 'Charger cet utilisateur'."
    )
elif page == "Ledger":
    # En-tête moderne pour la page Ledger
    st.markdown("""
        <div style="background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%); 
                    padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
            <h1 style="color: white; margin: 0; text-align: center;">
                💰 Gestion du Ledger
            </h1>
            <p style="color: #E8F5F2; margin: 0.5rem 0 0 0; text-align: center; font-size: 1.1rem;">
                Vue d'ensemble de vos comptes et opérations financières
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Section des comptes avec design amélioré
        st.markdown("#### 🏦 Vos Comptes")
        balances_by_type = compute_balances_by_type()

        # Calculer les totaux d'abord
        linked_total = sum(balances_by_type["linked"].values()) / 100
        unlinked_total = sum(balances_by_type["unlinked"].values()) / 100
        grand_total = linked_total + unlinked_total

        # Affichage des totaux en cartes colorées
        col_t1, col_t2, col_t3 = st.columns(3)
        
        with col_t1:
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           padding: 1rem; border-radius: 8px; color: white; text-align: center;">
                    <h4 style="margin: 0; font-size: 0.9rem;">🔗 LINKED</h4>
                    <h2 style="margin: 0.5rem 0; font-size: 1.5rem;">{linked_total:,.2f} €</h2>
                </div>
            """, unsafe_allow_html=True)
            
        with col_t2:
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                           padding: 1rem; border-radius: 8px; color: white; text-align: center;">
                    <h4 style="margin: 0; font-size: 0.9rem;">📎 UNLINKED</h4>
                    <h2 style="margin: 0.5rem 0; font-size: 1.5rem;">{unlinked_total:,.2f} €</h2>
                </div>
            """, unsafe_allow_html=True)
            
        with col_t3:
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                           padding: 1rem; border-radius: 8px; color: white; text-align: center;">
                    <h4 style="margin: 0; font-size: 0.9rem;">💰 TOTAL</h4>
                    <h2 style="margin: 0.5rem 0; font-size: 1.5rem;">{grand_total:,.2f} €</h2>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Liste détaillée des comptes avec style amélioré
        st.markdown("##### 📋 Détail des comptes")
        
        for acc in st.session_state.accounts:
            balance_eur = (
                balances_by_type[
                    "linked" if not acc.get("is_unlinked", False) else "unlinked"
                ].get(acc["account_id"], 0)
                / 100
            )
            
            # Déterminer la couleur selon le type et le solde
            if acc.get("is_unlinked", False):
                bg_color = "#fff5f5" if balance_eur >= 0 else "#ffe5e5"
                border_color = "#f093fb"
                icon = "📎"
                type_label = "Unlinked"
            else:
                bg_color = "#f0f8ff" if balance_eur >= 0 else "#ffe5e5"
                border_color = "#667eea" 
                icon = "🔗"
                type_label = "Linked"
            
            # Format du montant avec couleur
            amount_color = "#28a745" if balance_eur >= 0 else "#dc3545"
            
            st.markdown(f"""
                <div style="background: {bg_color}; 
                           border-left: 4px solid {border_color}; 
                           padding: 1rem; margin: 0.5rem 0; 
                           border-radius: 0 8px 8px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4 style="margin: 0; color: #333;">{icon} {acc['name']}</h4>
                            <small style="color: #666;">Type: {type_label}</small>
                        </div>
                        <div style="text-align: right;">
                            <h3 style="margin: 0; color: {amount_color}; font-weight: bold;">
                                {balance_eur:,.2f} €
                            </h3>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Section de suppression avec style amélioré
        if st.session_state.ledger:
            st.markdown("---")
            
            last = st.session_state.ledger[-1]
            try:
                acc_id_to_name = {
                    a["account_id"]: a["name"] for a in st.session_state.accounts
                }
                t = last.get("type")
                amt = last.get("amount_cents", 0) / 100
                src = acc_id_to_name.get(last.get("src_account_id", ""), "")
                dest = acc_id_to_name.get(last.get("dest_account_id", ""), "")
                accn = acc_id_to_name.get(last.get("account_id", ""), "")
                
                # Formatage des détails de la dernière opération
                details = []
                if src:
                    details.append(f"depuis {src}")
                if dest:
                    details.append(f"vers {dest}")
                if accn:
                    details.append(f"compte {accn}")
                
                meta = " | ".join([t, f"{amt:.2f}€"] + details)
                
                # Affichage stylé de la dernière opération
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); 
                               padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                        <h5 style="color: white; margin: 0;">🗑️ Dernière opération</h5>
                        <p style="color: #ffebee; margin: 0.5rem 0; font-size: 0.9rem;">
                            {last.get('ts','')}
                        </p>
                        <p style="color: white; margin: 0; font-weight: bold;">
                            {meta}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception:
                st.markdown("""
                    <div style="background: #ff6b6b; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                        <h5 style="color: white; margin: 0;">🗑️ Dernière opération</h5>
                        <p style="color: white; margin: 0.5rem 0;">Informations non disponibles</p>
                    </div>
                """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🗑️ Supprimer la dernière", key="delete_last_op", type="secondary"):
                    if delete_last_ledger_entry(st.session_state.pseudo):
                        # Recharger le ledger depuis la DB
                        st.session_state.ledger = load_user_ledger(st.session_state.pseudo)
                        st.success("✅ Dernière opération supprimée")
                        st.rerun()
                    else:
                        st.warning("⚠️ Aucune opération à supprimer")

        # Section des règles avec style amélioré


    with col_right:
        # Section d'ajout d'opération avec style amélioré
        st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                <h4 style="color: white; margin: 0; text-align: center;">
                    ✏️ Nouvelle Opération
                </h4>
            </div>
        """, unsafe_allow_html=True)
        op_type = st.selectbox(
            "Type",
            ["expense", "deposit", "transfer", "adjustment"],
            format_func=lambda t: {
                "deposit": "Dépôt",
                "expense": "Dépense",
                "transfer": "Transfert",
                "adjustment": "Ajustement",
            }[t],
        )

        # Sélecteurs de comptes selon le type
        acc_map_name_to_id = {
            acc["name"]: acc["account_id"] for acc in st.session_state.accounts
        }
        acc_names = list(acc_map_name_to_id.keys())

        src_id: Optional[str] = None
        dest_id: Optional[str] = None
        adj_id: Optional[str] = None

        if op_type == "deposit":
            dest_name = st.selectbox("Vers", acc_names)
            dest_id = acc_map_name_to_id[dest_name]
        elif op_type == "expense":
            src_name = st.selectbox("Depuis", acc_names)
            src_id = acc_map_name_to_id[src_name]
        elif op_type == "transfer":
            c1, c2 = st.columns(2)
            with c1:
                src_name = st.selectbox("Depuis", acc_names, key="tx_src")
                src_id = acc_map_name_to_id[src_name]
            with c2:
                dest_name = st.selectbox(
                    "Vers",
                    [n for n in acc_names if acc_map_name_to_id[n] != src_id],
                    key="tx_dest",
                )
                dest_id = acc_map_name_to_id[dest_name]
        elif op_type == "adjustment":
            adj_name = st.selectbox("Compte", acc_names)
            adj_id = acc_map_name_to_id[adj_name]

        amount_eur = st.number_input(
            "Montant (€)",
            step=0.01,
            format="%.2f",
            help="Toujours en euros; stocké en cents",
        )
        note = st.text_input("Note", placeholder="Commentaire…")

        if st.button("Ajouter l'opération", type="primary"):
            if op_type != "adjustment" and amount_eur <= 0:
                st.error("Montant strictement positif requis")
            else:
                amt_cents = int(round(amount_eur * 100))
                base = {"ts": now_iso_utc(), "type": op_type, "note": note}
                entry = None
                if op_type == "deposit":
                    entry = {
                        **base,
                        "amount_cents": amt_cents,
                        "dest_account_id": dest_id,
                    }
                elif op_type == "expense":
                    entry = {
                        **base,
                        "amount_cents": amt_cents,
                        "src_account_id": src_id,
                    }
                elif op_type == "transfer":
                    if src_id == dest_id:
                        st.error("Comptes source et destinataire différents requis")
                    else:
                        entry = {
                            **base,
                            "amount_cents": amt_cents,
                            "src_account_id": src_id,
                            "dest_account_id": dest_id,
                        }
                elif op_type == "adjustment":
                    # Ajustement autorise les montants signés
                    entry = {**base, "amount_cents": amt_cents, "account_id": adj_id}
                if entry is not None:
                    save_ledger_entry(st.session_state.pseudo, entry)
                    st.session_state.ledger.append(entry)
                    st.success("Opération ajoutée")
                    st.rerun()

        if st.session_state.rules:
            st.markdown("---")
            st.markdown("""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                           padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                    <h4 style="color: white; margin: 0; text-align: center;">
                        ⚡ Règles Rapides
                    </h4>
                </div>
            """, unsafe_allow_html=True)
            for rule in st.session_state.rules:
                cols = (
                    st.columns([2, 1]) if rule.get("require_value") else st.columns([3])
                )
                with cols[0]:
                    value_input = None
                    if rule.get("require_value"):
                        label_text = f"Valeur (€) — {rule['name']}"
                        if rule.get("use_balance_difference"):
                            balances_by_type = compute_balances_by_type()
                            current_linked_total = (
                                sum(balances_by_type["linked"].values()) / 100.0
                            )
                            label_text = f"Valeur cible (€) — {rule['name']} [Solde linked: {current_linked_total:.2f}€]"
                        value_input = st.number_input(
                            label_text,
                            key=f"rule_val_{rule['rule_id']}",
                            min_value=0.0,
                            step=0.01,
                            format="%.2f",
                        )

                        # Afficher l'aide en temps réel quand une valeur est saisie
                        if value_input and value_input > 0:
                            if rule.get("use_balance_difference", False):
                                # Mode différence de solde (uniquement comptes linked)
                                balances_by_type = compute_balances_by_type()
                                current_linked_total_cents = sum(
                                    balances_by_type["linked"].values()
                                )
                                current_linked_total_eur = (
                                    current_linked_total_cents / 100.0
                                )
                                current_unlinked_total_eur = (
                                    sum(balances_by_type["unlinked"].values()) / 100.0
                                )
                                target_value_eur = value_input
                                difference_eur = (
                                    target_value_eur - current_linked_total_eur
                                )

                                if abs(difference_eur) < 0.01:
                                    st.info(
                                        f"📊 Solde linked: {current_linked_total_eur:.2f}€, Cible: {target_value_eur:.2f}€ → Aucun ajustement nécessaire"
                                    )
                                else:
                                    with st.expander(
                                        f"📊 Aperçu: Différence à répartir {difference_eur:+.2f}€",
                                        expanded=True,
                                    ):
                                        st.write(
                                            f"**🔗 Solde linked:** {current_linked_total_eur:.2f}€"
                                        )
                                        st.write(
                                            f"**📎 Solde unlinked:** {current_unlinked_total_eur:.2f}€"
                                        )
                                        st.write(
                                            f"**🎯 Valeur cible (linked):** {target_value_eur:.2f}€"
                                        )
                                        st.write(
                                            f"**📊 Différence à répartir:** {difference_eur:+.2f}€"
                                        )
                            else:
                                # Mode normal
                                with st.expander(
                                    f"💰 Aperçu: Montant à répartir {value_input:.2f}€",
                                    expanded=True,
                                ):
                                    st.write(f"**Valeur saisie:** {value_input:.2f}€")
                with cols[1] if rule.get("require_value") else cols[0]:
                    btn_label = rule.get("trigger_label") or f"Exécuter: {rule['name']}"
                    if st.button(btn_label, key=f"exec_{rule['rule_id']}"):
                        # Déterminer la base (cents)
                        if rule.get("use_balance_difference", False) and rule.get(
                            "require_value"
                        ):
                            # Mode différence de solde : calculer la différence entre la valeur saisie et le solde linked uniquement
                            balances_by_type = compute_balances_by_type()
                            current_linked_total_cents = sum(
                                balances_by_type["linked"].values()
                            )
                            target_value_cents = int(round((value_input or 0.0) * 100))
                            base_value_cents = (
                                target_value_cents - current_linked_total_cents
                            )

                            # Vérifier si il y a une différence significative
                            if (
                                abs(base_value_cents) < 1
                            ):  # Moins d'1 centime de différence
                                st.warning(
                                    "Aucun ajustement nécessaire (différence < 0.01€)"
                                )
                                continue
                        else:
                            # Mode normal : utiliser directement la valeur saisie ou par défaut
                            base_value_cents = (
                                int(round((value_input or 0.0) * 100))
                                if rule.get("require_value")
                                else int(rule.get("default_amount_cents", 0))
                            )
                        actions = rule.get("actions", [])
                        # 1) Appliquer les montants fixes en premier (somme des fixes)
                        fixed_total = sum(
                            max(0, int(a.get("fixed_cents", 0))) for a in actions
                        )
                        # 2) Calculer la base restante pour les pourcentages
                        remaining_base = max(base_value_cents - fixed_total, 0)

                        # Appliquer les actions
                        ts = now_iso_utc()
                        for act in actions:
                            kind = act.get(
                                "kind"
                            )  # deposit/expense/transfer/adjustment
                            percent = float(act.get("percent", 0.0))
                            fixed = int(act.get("fixed_cents", 0))
                            # Variable sur la base restante
                            variable_part = (
                                int(math.floor(remaining_base * percent))
                                if percent > 0
                                else 0
                            )
                            amount = max(0, fixed) + max(0, variable_part)
                            note = f"Rule:{rule['name']}"
                            entry = None
                            if kind == "deposit":
                                entry = {
                                    "ts": ts,
                                    "type": "deposit",
                                    "amount_cents": amount,
                                    "dest_account_id": act.get("dest_account_id"),
                                    "note": note,
                                }
                            elif kind == "expense":
                                entry = {
                                    "ts": ts,
                                    "type": "expense",
                                    "amount_cents": amount,
                                    "src_account_id": act.get("src_account_id"),
                                    "note": note,
                                }
                            elif kind == "transfer":
                                src_a = act.get("src_account_id")
                                dest_a = act.get("dest_account_id")
                                if src_a == dest_a:
                                    # ignorer transferts invalides (src == dest)
                                    entry = None
                                else:
                                    entry = {
                                        "ts": ts,
                                        "type": "transfer",
                                        "amount_cents": amount,
                                        "src_account_id": src_a,
                                        "dest_account_id": dest_a,
                                        "note": note,
                                    }
                            elif kind == "adjustment":
                                entry = {
                                    "ts": ts,
                                    "type": "adjustment",
                                    "amount_cents": amount,
                                    "account_id": act.get("account_id"),
                                    "note": note,
                                }
                            if entry is not None:
                                save_ledger_entry(st.session_state.pseudo, entry)
                                st.session_state.ledger.append(entry)
                        st.success(f"Règle '{rule['name']}' exécutée")
                        st.rerun()
    # Journal des opérations avec design amélioré
    st.markdown("---")
    st.markdown("""
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                   padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <h4 style="color: white; margin: 0; text-align: center;">
                📋 Journal des Opérations
            </h4>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.ledger:
        acc_id_to_name = {
            acc["account_id"]: acc["name"] for acc in st.session_state.accounts
        }
        
        # Limiter l'affichage aux 10 dernières opérations pour éviter la surcharge
        recent_ops = list(reversed(st.session_state.ledger))[:10]
        
        for i, op in enumerate(recent_ops):
            t = op.get("type", "")
            amt = op.get("amount_cents", 0) / 100
            ts = op.get("ts", "")
            note = op.get("note", "")
            
            # Icônes et couleurs selon le type
            type_config = {
                "deposit": {"icon": "💰", "color": "#28a745", "label": "Dépôt"},
                "expense": {"icon": "💸", "color": "#dc3545", "label": "Dépense"},
                "transfer": {"icon": "🔄", "color": "#007bff", "label": "Transfert"},
                "adjustment": {"icon": "⚖️", "color": "#ffc107", "label": "Ajustement"}
            }
            
            config = type_config.get(t, {"icon": "❓", "color": "#6c757d", "label": t})
            
            # Construction du texte détaillé
            details = []
            if op.get("src_account_id"):
                details.append(f"Depuis: {acc_id_to_name.get(op.get('src_account_id'), 'N/A')}")
            if op.get("dest_account_id"):
                details.append(f"Vers: {acc_id_to_name.get(op.get('dest_account_id'), 'N/A')}")
            if op.get("account_id"):
                details.append(f"Compte: {acc_id_to_name.get(op.get('account_id'), 'N/A')}")
            
            detail_text = " | ".join(details) if details else ""
            
            # Formatage de la date
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                formatted_date = dt.strftime("%d/%m/%Y %H:%M")
            except:
                formatted_date = ts
            
            # Construire les parties conditionnelles
            detail_html = ""
            if detail_text:
                detail_html = f'<div style="color: #555; font-size: 0.9rem; margin-bottom: 0.3rem;">{detail_text}</div>'
            
            note_html = ""
            if note:
                note_html = f'<div style="color: #777; font-style: italic; font-size: 0.85rem;">💬 {note}</div>'
            
            st.markdown(f"""
                <div style="background: white; border-left: 4px solid {config['color']}; 
                           padding: 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0;
                           box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1;">
                            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                                <span style="font-size: 1.2rem; margin-right: 0.5rem;">{config['icon']}</span>
                                <strong style="color: {config['color']};">{config['label']}</strong>
                                <span style="margin-left: 1rem; color: #666; font-size: 0.9rem;">{formatted_date}</span>
                            </div>
                            {detail_html}
                            {note_html}
                        </div>
                        <div style="text-align: right; margin-left: 1rem;">
                            <span style="font-size: 1.3rem; font-weight: bold; color: {config['color']};">
                                {amt:+.2f} €
                            </span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Si il y a plus de 10 opérations, proposer de voir toutes via un expander
        if len(st.session_state.ledger) > 10:
            with st.expander(f"📊 Voir toutes les opérations ({len(st.session_state.ledger)} au total)"):
                rows = []
                for op in reversed(st.session_state.ledger):
                    t = op.get("type")
                    amt = op.get("amount_cents", 0) / 100
                    rows.append({
                        "Date (UTC)": op.get("ts"),
                        "Type": t,
                        "Depuis": acc_id_to_name.get(op.get("src_account_id", ""), ""),
                        "Vers": acc_id_to_name.get(op.get("dest_account_id", ""), ""),
                        "Compte": acc_id_to_name.get(op.get("account_id", ""), ""),
                        "Montant (€)": f"{amt:.2f}",
                        "Note": op.get("note", ""),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.markdown("""
            <div style="background: #f8f9fa; padding: 2rem; border-radius: 10px; text-align: center;">
                <span style="font-size: 3rem;">📝</span>
                <h4 style="color: #6c757d; margin: 1rem 0;">Aucune opération</h4>
                <p style="color: #868e96;">Commencez par ajouter votre première opération !</p>
            </div>
        """, unsafe_allow_html=True)

elif page == "Règles":
    st.header("Règles et Comptes")

    # Gestion des comptes (déplacée ici)
    st.subheader("Comptes")
    balances = compute_balances()
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "ID": acc["account_id"],
                    "Nom": acc["name"],
                    "Solde (€)": f"{balances.get(acc['account_id'], 0)/100:.2f}",
                }
                for acc in st.session_state.accounts
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    with st.form("create_account_form_rules", clear_on_submit=True):
        new_name = st.text_input("Nom du compte", placeholder="Mon compte")
        new_id = st.text_input(
            "ID du compte",
            placeholder="mon_compte",
            help="Identifiant unique (lettres, chiffres, tirets, underscores)",
        )
        is_unlinked = st.checkbox(
            "Compte unlinked",
            value=False,
            help="Les comptes unlinked ne sont pas pris en compte dans les calculs de différence des règles",
        )
        submitted = st.form_submit_button("Créer")
        if submitted:
            if not new_name or not new_id:
                st.error("Nom et ID requis")
            elif not re.match(r"^[a-zA-Z0-9_-]+$", new_id):
                st.error("ID invalide")
            elif account_index(new_id) is not None:
                st.error("ID déjà existant")
            else:
                save_account(st.session_state.pseudo, new_id, new_name, is_unlinked)
                st.session_state.accounts.append(
                    {"account_id": new_id, "name": new_name, "is_unlinked": is_unlinked}
                )
                st.success("Compte créé")
                st.rerun()

    del_id = st.selectbox(
        "Supprimer un compte",
        options=[acc["account_id"] for acc in st.session_state.accounts],
        format_func=lambda i: next(
            a["name"] for a in st.session_state.accounts if a["account_id"] == i
        ),
    )
    if st.button("Supprimer le compte", type="secondary"):
        if del_id and balances.get(del_id, 0) != 0:
            st.error("Solde non nul")
        elif del_id:
            delete_account(st.session_state.pseudo, del_id)
            idx = account_index(del_id)
            if idx is not None:
                st.session_state.accounts.pop(idx)
            st.success("Compte supprimé")
            st.rerun()

    st.markdown("---")
    st.subheader("Règles")

    # Information sur le mode différence de solde
    with st.expander("ℹ️ Aide - Mode différence de solde", expanded=False):
        st.markdown(
            """
        **Mode différence de solde** 📊
        
        Lorsque cette option est activée pour une règle :
        - Au lieu d'appliquer la règle sur la valeur saisie directement
        - La règle s'applique sur la **différence** entre la valeur saisie et le solde total actuel
        
        **Exemple pratique :**
        - Solde total actuel : 1000€
        - Vous saisissez : 1200€ (votre objectif)  
        - Différence calculée : 200€ (1200€ - 1000€)
        - La règle s'applique sur ces 200€ de différence
        
        **Cas d'usage :** Idéal pour ajuster votre répartition vers un objectif cible sans refaire tous les calculs manuellement.
        """
        )

    # Editeur de nouvelle règle
    with st.expander("Créer une règle", expanded=True):
        rule_name = st.text_input("Nom de la règle", placeholder="Salaire")
        require_value = st.checkbox("Demander une valeur à l'exécution", value=True)
        use_balance_difference = False
        if require_value:
            use_balance_difference = st.checkbox(
                "Appliquer sur la différence de solde",
                value=False,
                help="Si coché, la règle sera appliquée sur la différence entre la valeur saisie et le solde total actuel",
            )
        default_amount_eur = 0.0
        if not require_value:
            default_amount_eur = st.number_input(
                "Montant par défaut (€)", min_value=0.0, step=0.01, format="%.2f"
            )
        trigger_label = st.text_input("Libellé du bouton (trigger)", value="Exécuter")

        # Actions de la règle
        acc_map = {acc["name"]: acc["account_id"] for acc in st.session_state.accounts}
        acc_names = list(acc_map.keys())

        st.caption(
            "Actions: chaque action calcule un montant depuis la valeur de base (ou un fixe)"
        )
        for i, act in enumerate(st.session_state.new_rule_actions):
            with st.container():
                st.write(f"Action #{i+1}")
                c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 0.6])
                with c1:
                    kind = st.selectbox(
                        "Type",
                        ["deposit", "expense", "transfer", "adjustment"],
                        index=["deposit", "expense", "transfer", "adjustment"].index(
                            act.get("kind", "deposit")
                        ),
                        key=f"act_kind_{i}",
                    )
                    act["kind"] = kind
                with c2:
                    percent = st.number_input(
                        "% base",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(act.get("percent", 0.0)) * 100.0,
                        step=1.0,
                        key=f"act_pct_{i}",
                        help="0..100%. Si 0 et 'fixe' >0, on utilisera le fixe.",
                    )
                    act["percent"] = percent / 100.0
                with c3:
                    fixed = st.number_input(
                        "Fixe (€)",
                        min_value=0.0,
                        step=0.01,
                        format="%.2f",
                        value=(act.get("fixed_cents", 0) / 100.0),
                        key=f"act_fix_{i}",
                    )
                    act["fixed_cents"] = int(round(fixed * 100))
                with c4:
                    if st.button("Suppr", key=f"del_act_{i}"):
                        st.session_state.new_rule_actions.pop(i)
                        st.rerun()
                # Sélecteurs de comptes selon type
                if kind in ("deposit", "transfer"):
                    dest = st.selectbox(
                        "Vers",
                        acc_names,
                        index=max(
                            0,
                            acc_names.index(
                                next(
                                    (
                                        n
                                        for n, a in acc_map.items()
                                        if a == act.get("dest_account_id")
                                    ),
                                    acc_names[0],
                                )
                            ),
                        ),
                        key=f"act_dest_{i}",
                    )
                    act["dest_account_id"] = acc_map[dest]
                if kind in ("expense", "transfer"):
                    src = st.selectbox(
                        "Depuis",
                        acc_names,
                        index=max(
                            0,
                            acc_names.index(
                                next(
                                    (
                                        n
                                        for n, a in acc_map.items()
                                        if a == act.get("src_account_id")
                                    ),
                                    acc_names[0],
                                )
                            ),
                        ),
                        key=f"act_src_{i}",
                    )
                    act["src_account_id"] = acc_map[src]
                if kind == "adjustment":
                    accn = st.selectbox(
                        "Compte",
                        acc_names,
                        index=max(
                            0,
                            acc_names.index(
                                next(
                                    (
                                        n
                                        for n, a in acc_map.items()
                                        if a == act.get("account_id")
                                    ),
                                    acc_names[0],
                                )
                            ),
                        ),
                        key=f"act_acc_{i}",
                    )
                    act["account_id"] = acc_map[accn]
        if st.button("Ajouter une action"):
            st.session_state.new_rule_actions.append(
                {"kind": "deposit", "percent": 0.0, "fixed_cents": 0}
            )
            st.rerun()

        if st.button("Enregistrer la règle", type="primary"):
            if not rule_name:
                st.error("Nom requis")
            elif (
                require_value is False
                and default_amount_eur <= 0
                and not st.session_state.new_rule_actions
            ):
                st.error("Règle vide")
            else:
                new_rule = {
                    "rule_id": f"r{len(st.session_state.rules)+1}",
                    "name": rule_name,
                    "require_value": require_value,
                    "default_amount_cents": int(round(default_amount_eur * 100)),
                    "trigger_label": trigger_label or "Exécuter",
                    "use_balance_difference": use_balance_difference,
                    "actions": st.session_state.new_rule_actions.copy(),
                }
                save_rule(st.session_state.pseudo, new_rule)
                st.session_state.rules.append(new_rule)
                st.session_state.new_rule_actions = []
                st.success("Règle créée")
                st.rerun()

    # Éditeur de règle existante
    if st.session_state.editing_rule_id:
        # Trouver la règle en cours d'édition
        editing_rule = None
        for rule in st.session_state.rules:
            if rule["rule_id"] == st.session_state.editing_rule_id:
                editing_rule = rule
                break

        if editing_rule:
            with st.expander(
                f"✏️ Modifier la règle: {editing_rule['name']}", expanded=True
            ):
                st.info("Mode édition activé - modifiez les paramètres ci-dessous")

                # Formulaire d'édition
                edit_rule_name = st.text_input(
                    "Nom de la règle",
                    value=editing_rule.get("name", ""),
                    key="edit_rule_name",
                )
                edit_require_value = st.checkbox(
                    "Demander une valeur à l'exécution",
                    value=editing_rule.get("require_value", True),
                    key="edit_require_value",
                )
                edit_use_balance_difference = False
                if edit_require_value:
                    edit_use_balance_difference = st.checkbox(
                        "Appliquer sur la différence de solde",
                        value=editing_rule.get("use_balance_difference", False),
                        help="Si coché, la règle sera appliquée sur la différence entre la valeur saisie et le solde total actuel",
                        key="edit_use_balance_difference",
                    )
                edit_default_amount_eur = 0.0
                if not edit_require_value:
                    edit_default_amount_eur = st.number_input(
                        "Montant par défaut (€)",
                        min_value=0.0,
                        step=0.01,
                        format="%.2f",
                        value=editing_rule.get("default_amount_cents", 0) / 100.0,
                        key="edit_default_amount",
                    )
                edit_trigger_label = st.text_input(
                    "Libellé du bouton (trigger)",
                    value=editing_rule.get("trigger_label", "Exécuter"),
                    key="edit_trigger_label",
                )

                # Actions de la règle en cours d'édition
                acc_map = {
                    acc["name"]: acc["account_id"] for acc in st.session_state.accounts
                }
                acc_names = list(acc_map.keys())

                st.caption("Actions: modifiez les actions existantes")
                for i, act in enumerate(st.session_state.edit_rule_actions):
                    with st.container():
                        st.write(f"Action #{i+1}")
                        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 0.6])
                        with c1:
                            kind = st.selectbox(
                                "Type",
                                ["deposit", "expense", "transfer", "adjustment"],
                                index=[
                                    "deposit",
                                    "expense",
                                    "transfer",
                                    "adjustment",
                                ].index(act.get("kind", "deposit")),
                                key=f"edit_act_kind_{i}",
                            )
                            act["kind"] = kind
                        with c2:
                            percent = st.number_input(
                                "% base",
                                min_value=0.0,
                                max_value=100.0,
                                value=float(act.get("percent", 0.0)) * 100.0,
                                step=1.0,
                                key=f"edit_act_pct_{i}",
                                help="0..100%. Si 0 et 'fixe' >0, on utilisera le fixe.",
                            )
                            act["percent"] = percent / 100.0
                        with c3:
                            fixed = st.number_input(
                                "Fixe (€)",
                                min_value=0.0,
                                step=0.01,
                                format="%.2f",
                                value=(act.get("fixed_cents", 0) / 100.0),
                                key=f"edit_act_fix_{i}",
                            )
                            act["fixed_cents"] = int(round(fixed * 100))
                        with c4:
                            if st.button("Suppr", key=f"edit_del_act_{i}"):
                                st.session_state.edit_rule_actions.pop(i)
                                st.rerun()

                        # Sélecteurs de comptes selon type
                        if kind in ("deposit", "transfer"):
                            dest = st.selectbox(
                                "Vers",
                                acc_names,
                                index=max(
                                    0,
                                    (
                                        acc_names.index(
                                            next(
                                                (
                                                    n
                                                    for n, a in acc_map.items()
                                                    if a == act.get("dest_account_id")
                                                ),
                                                acc_names[0] if acc_names else "",
                                            )
                                        )
                                        if acc_names
                                        else 0
                                    ),
                                ),
                                key=f"edit_act_dest_{i}",
                            )
                            act["dest_account_id"] = acc_map[dest]
                        if kind in ("expense", "transfer"):
                            src = st.selectbox(
                                "Depuis",
                                acc_names,
                                index=max(
                                    0,
                                    (
                                        acc_names.index(
                                            next(
                                                (
                                                    n
                                                    for n, a in acc_map.items()
                                                    if a == act.get("src_account_id")
                                                ),
                                                acc_names[0] if acc_names else "",
                                            )
                                        )
                                        if acc_names
                                        else 0
                                    ),
                                ),
                                key=f"edit_act_src_{i}",
                            )
                            act["src_account_id"] = acc_map[src]
                        if kind == "adjustment":
                            accn = st.selectbox(
                                "Compte",
                                acc_names,
                                index=max(
                                    0,
                                    (
                                        acc_names.index(
                                            next(
                                                (
                                                    n
                                                    for n, a in acc_map.items()
                                                    if a == act.get("account_id")
                                                ),
                                                acc_names[0] if acc_names else "",
                                            )
                                        )
                                        if acc_names
                                        else 0
                                    ),
                                ),
                                key=f"edit_act_acc_{i}",
                            )
                            act["account_id"] = acc_map[accn]

                if st.button("Ajouter une action", key="edit_add_action"):
                    st.session_state.edit_rule_actions.append(
                        {"kind": "deposit", "percent": 0.0, "fixed_cents": 0}
                    )
                    st.rerun()

                # Boutons de sauvegarde et annulation
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "💾 Sauvegarder les modifications",
                        type="primary",
                        key="save_edit_rule",
                    ):
                        if not edit_rule_name:
                            st.error("Nom requis")
                        else:
                            # Mettre à jour la règle
                            updated_rule = {
                                "rule_id": editing_rule["rule_id"],
                                "name": edit_rule_name,
                                "require_value": edit_require_value,
                                "default_amount_cents": int(
                                    round(edit_default_amount_eur * 100)
                                ),
                                "trigger_label": edit_trigger_label or "Exécuter",
                                "use_balance_difference": edit_use_balance_difference,
                                "actions": st.session_state.edit_rule_actions.copy(),
                            }

                            # Sauvegarder en base
                            update_rule(st.session_state.pseudo, updated_rule)

                            # Mettre à jour la liste en mémoire
                            for i, rule in enumerate(st.session_state.rules):
                                if rule["rule_id"] == editing_rule["rule_id"]:
                                    st.session_state.rules[i] = updated_rule
                                    break

                            # Sortir du mode édition
                            st.session_state.editing_rule_id = None
                            st.session_state.edit_rule_actions = []

                            st.success("Règle modifiée avec succès!")
                            st.rerun()

                with col2:
                    if st.button("❌ Annuler", key="cancel_edit_rule"):
                        st.session_state.editing_rule_id = None
                        st.session_state.edit_rule_actions = []
                        st.rerun()

    # Liste des règles existantes
    if st.session_state.rules:
        st.subheader("Règles existantes")
        for idx, rule in enumerate(st.session_state.rules):
            with st.expander(f"{rule['name']} ({rule['rule_id']})"):
                st.write(
                    f"• Demande valeur: {'Oui' if rule.get('require_value') else 'Non'}"
                )
                if rule.get("require_value") and rule.get("use_balance_difference"):
                    st.write("• **Mode différence de solde activé** 📊")
                st.write(
                    f"• Montant par défaut: {rule.get('default_amount_cents',0)/100:.2f} €"
                )
                st.write(f"• Bouton: {rule.get('trigger_label','Exécuter')}")
                if rule.get("actions"):
                    st.write("• Actions:")
                    for a in rule["actions"]:
                        desc = a.get("kind")
                        if a.get("percent"):
                            desc += f" {a['percent']*100:.0f}%"
                        if a.get("fixed_cents"):
                            desc += f" + {a['fixed_cents']/100:.2f}€"
                        if a.get("src_account_id"):
                            desc += f" | src={a['src_account_id']}"
                        if a.get("dest_account_id"):
                            desc += f" | dest={a['dest_account_id']}"
                        if a.get("account_id"):
                            desc += f" | acc={a['account_id']}"
                        st.write(f"- {desc}")
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("Modifier", key=f"edit_rule_{idx}"):
                        # Activer l'édition de cette règle
                        st.session_state.editing_rule_id = rule["rule_id"]
                        st.session_state.edit_rule_actions = [
                            a.copy() for a in rule.get("actions", [])
                        ]
                        st.rerun()
                with c2:
                    if st.button("Supprimer cette règle", key=f"del_rule_{idx}"):
                        rule_id_to_delete = st.session_state.rules[idx]["rule_id"]
                        delete_rule(st.session_state.pseudo, rule_id_to_delete)
                        st.session_state.rules.pop(idx)
                        st.rerun()
                with c3:
                    # Dupliquer
                    if st.button("Dupliquer", key=f"dup_rule_{idx}"):
                        dup = rule.copy()
                        dup["rule_id"] = f"r{len(st.session_state.rules)+1}"
                        dup["name"] = rule["name"] + " (copie)"
                        dup["actions"] = [a.copy() for a in rule.get("actions", [])]
                        save_rule(st.session_state.pseudo, dup)
                        st.session_state.rules.append(dup)
                        st.rerun()

elif page == "📊 Dashboard":
    # En-tête moderne avec gradient visuel
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
            <h1 style="color: white; margin: 0; text-align: center;">
                📊 Dashboard Financier
            </h1>
            <p style="color: #E8E8E8; margin: 0.5rem 0 0 0; text-align: center; font-size: 1.1rem;">
                Analyses avancées et visualisations de vos finances
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    if not st.session_state.pseudo:
        st.warning("🔐 Sélectionnez un utilisateur pour accéder au dashboard.")
    elif not st.session_state.ledger:
        st.info(
            "📭 Aucune donnée disponible. Ajoutez des opérations pour voir les analyses."
        )
    else:
        # Sidebar avec filtres améliorés
        st.sidebar.markdown(
            """
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <h3 style="margin: 0; color: #495057;">🔧 Configuration</h3>
            </div>
        """,
            unsafe_allow_html=True,
        )

        # Séparation des comptes linked/unlinked
        linked_accounts = get_linked_accounts()
        unlinked_accounts = get_unlinked_accounts()

        # Options d'analyse par type de compte
        st.sidebar.subheader("📊 Scope d'analyse")
        analysis_scope = st.sidebar.radio(
            "Comptes à inclure",
            ["🔗 Linked uniquement", "📎 Unlinked uniquement", "💰 Tous les comptes"],
            index=0,
        )

        # Sélection des comptes selon le scope
        if analysis_scope == "🔗 Linked uniquement":
            available_accounts = linked_accounts
        elif analysis_scope == "📎 Unlinked uniquement":
            available_accounts = unlinked_accounts
        else:
            available_accounts = st.session_state.accounts

        account_options = ["Tous"] + [acc["name"] for acc in available_accounts]
        selected_accounts = st.sidebar.multiselect(
            "Comptes spécifiques", account_options, default=["Tous"]
        )

        # Convertir la sélection
        if "Tous" in selected_accounts:
            selected_account_ids = [acc["account_id"] for acc in available_accounts]
        else:
            selected_account_ids = [
                acc["account_id"]
                for acc in available_accounts
                if acc["name"] in selected_accounts
            ]

        # Période d'analyse avec plus d'options
        st.sidebar.subheader("📅 Période")
        period_days = st.sidebar.selectbox(
            "Durée d'analyse",
            [7, 15, 30, 60, 90, 180, 365],
            index=2,
            format_func=lambda x: f"{x} jours" + (" (1 an)" if x == 365 else ""),
        )

        # Options avancées
        st.sidebar.subheader("⚙️ Options avancées")
        show_trends = st.sidebar.checkbox("Afficher les tendances", value=True)
        show_predictions = st.sidebar.checkbox("Activer les prédictions", value=True)
        prediction_days = (
            st.sidebar.slider("Horizon prédiction (j)", 7, 90, 30)
            if show_predictions
            else 0
        )

        # Paramètres visuels
        chart_theme = st.sidebar.selectbox(
            "Thème des graphiques", ["Moderne", "Classique", "Sombre"], index=0
        )

        # Calculer les KPIs selon le scope d'analyse
        if analysis_scope == "🔗 Linked uniquement":
            analysis_accounts = linked_accounts
        elif analysis_scope == "📎 Unlinked uniquement":
            analysis_accounts = unlinked_accounts
        else:
            analysis_accounts = st.session_state.accounts
            
        kpis = calculate_kpis(st.session_state.ledger, analysis_accounts)
        balances_by_type = compute_balances_by_type()

        # KPIs modernes avec cartes colorées
        st.markdown("### 💎 Vue d'ensemble financière")
        
        # Première ligne - Soldes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            linked_total = sum(balances_by_type["linked"].values()) / 100
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           padding: 1.5rem; border-radius: 10px; color: white;">
                    <h3 style="margin: 0; font-size: 1.1rem;">🔗 Comptes Linked</h3>
                    <h1 style="margin: 0.5rem 0; font-size: 2rem;">{linked_total:,.2f} €</h1>
                    <p style="margin: 0; opacity: 0.8;">Comptes opérationnels</p>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            unlinked_total = sum(balances_by_type["unlinked"].values()) / 100
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                           padding: 1.5rem; border-radius: 10px; color: white;">
                    <h3 style="margin: 0; font-size: 1.1rem;">📎 Comptes Unlinked</h3>
                    <h1 style="margin: 0.5rem 0; font-size: 2rem;">{unlinked_total:,.2f} €</h1>
                    <p style="margin: 0; opacity: 0.8;">Épargnes & placements</p>
                </div>
            """, unsafe_allow_html=True)
            
        with col3:
            total_patrimoine = linked_total + unlinked_total
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                           padding: 1.5rem; border-radius: 10px; color: white;">
                    <h3 style="margin: 0; font-size: 1.1rem;">� Patrimoine Total</h3>
                    <h1 style="margin: 0.5rem 0; font-size: 2rem;">{total_patrimoine:,.2f} €</h1>
                    <p style="margin: 0; opacity: 0.8;">Valeur totale</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Deuxième ligne - Métriques de performance
        st.markdown("### 📊 Métriques de performance")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="� Tendance 7j",
                value=f"{kpis.get('tendance_7j', 0):+.2f} €",
                delta=f"{(kpis.get('tendance_7j', 0)/7):.2f} €/j" if kpis.get('tendance_7j', 0) != 0 else "Stable"
            )

        with col2:
            st.metric(
                label="� Dépenses moy/j",
                value=f"{kpis.get('depense_moy_jour', 0):.2f} €",
                delta=f"{kpis.get('depenses_30j', 0):.0f} € sur 30j"
            )

        with col3:
            st.metric(
                label="� Recettes 30j", 
                value=f"{kpis.get('recettes_30j', 0):.2f} €",
                delta=f"{(kpis.get('recettes_30j', 0) - kpis.get('depenses_30j', 0)):+.2f} € net"
            )

        with col4:
            volatility = abs(kpis.get('tendance_7j', 0)) / 7 if kpis.get('tendance_7j', 0) != 0 else 0
            st.metric(
                label="� Volatilité",
                value=f"{volatility:.2f} €/j",
                delta="Stable" if volatility < 10 else "Variable"
            )

        st.markdown("---")

        # Onglets améliorés avec plus d'analyses
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Évolution & Tendances",
            "🏦 Analyse par Compte", 
            "💰 Répartition & Structure",
            "💎 Simulations d'Intérêts",
            "🔮 Prédictions & IA",
            "🎯 Insights & Recommandations"
        ])

        with tab1:
            st.markdown("#### 📈 Évolution Temporelle")
            
            # Sélection du type d'analyse
            col_type, col_period = st.columns([2, 1])
            with col_type:
                view_type = st.selectbox(
                    "Type de vue",
                    ["Solde Global", "Linked vs Unlinked", "Flux Quotidiens", "Comparaison Périodes"]
                )
            with col_period:
                smoothing = st.selectbox("Lissage", ["Aucun", "Moyenne Mobile 7j", "Tendance"])

            # Récupération des données selon le scope
            if analysis_scope == "🔗 Linked uniquement":
                balance_evolution = get_balance_evolution(st.session_state.ledger, linked_accounts, period_days)
            elif analysis_scope == "📎 Unlinked uniquement":
                balance_evolution = get_balance_evolution(st.session_state.ledger, unlinked_accounts, period_days)
            else:
                balance_evolution = get_balance_evolution(st.session_state.ledger, st.session_state.accounts, period_days)

            if not balance_evolution.empty:
                # Configuration du thème selon la sélection
                if chart_theme == "Moderne":
                    color_palette = ["#667eea", "#764ba2", "#f093fb", "#f5576c"]
                elif chart_theme == "Sombre":
                    color_palette = ["#1e3c72", "#2a5298", "#833ab4", "#fd1d1d"]
                else:
                    color_palette = px.colors.qualitative.Set2

                if view_type == "Solde Global":
                    # Graphique d'évolution amélioré
                    fig_balance = px.area(
                        balance_evolution,
                        x="date",
                        y="total_euros",
                        title=f"💰 Évolution du patrimoine - {period_days} derniers jours",
                        labels={"total_euros": "Solde (€)", "date": "Date"},
                        color_discrete_sequence=color_palette
                    )
                    
                    # Ajouter une ligne de tendance si demandé
                    if smoothing == "Moyenne Mobile 7j" and len(balance_evolution) >= 7:
                        balance_evolution["ma_7"] = balance_evolution["total_euros"].rolling(7, min_periods=1).mean()
                        fig_balance.add_scatter(
                            x=balance_evolution["date"], 
                            y=balance_evolution["ma_7"],
                            mode='lines',
                            name='Tendance 7j',
                            line=dict(color='red', width=2, dash='dash')
                        )
                    
                    fig_balance.update_layout(
                        xaxis_title="📅 Date", 
                        yaxis_title="💰 Solde (€)", 
                        hovermode="x unified",
                        showlegend=True,
                        height=500
                    )
                    st.plotly_chart(fig_balance, width="stretch")
                    
                elif view_type == "Linked vs Unlinked":
                    # Évolution comparative linked vs unlinked
                    linked_evolution = get_balance_evolution(st.session_state.ledger, linked_accounts, period_days)
                    unlinked_evolution = get_balance_evolution(st.session_state.ledger, unlinked_accounts, period_days)
                    
                    fig_comp = go.Figure()
                    
                    fig_comp.add_trace(go.Scatter(
                        x=linked_evolution["date"],
                        y=linked_evolution["total_euros"],
                        mode='lines+markers',
                        name='🔗 Comptes Linked',
                        line=dict(color=color_palette[0], width=3),
                        fill='tonexty'
                    ))
                    
                    fig_comp.add_trace(go.Scatter(
                        x=unlinked_evolution["date"],
                        y=unlinked_evolution["total_euros"],
                        mode='lines+markers',
                        name='📎 Comptes Unlinked',
                        line=dict(color=color_palette[1], width=3),
                        fill='tozeroy'
                    ))
                    
                    fig_comp.update_layout(
                        title="🏦 Évolution Comparative: Linked vs Unlinked",
                        xaxis_title="📅 Date",
                        yaxis_title="💰 Solde (€)",
                        hovermode="x unified",
                        height=500
                    )
                    st.plotly_chart(fig_comp, width="stretch")
                    
                elif view_type == "Flux Quotidiens":
                    # Analyse des flux quotidiens
                    flows_df = get_daily_flow_analysis(st.session_state.ledger, period_days)
                    if not flows_df.empty:
                        fig_flows = go.Figure()
                        
                        fig_flows.add_trace(go.Bar(
                            x=flows_df["date"],
                            y=flows_df["recettes"],
                            name="💹 Recettes",
                            marker_color=color_palette[0],
                            opacity=0.8
                        ))
                        
                        fig_flows.add_trace(go.Bar(
                            x=flows_df["date"],
                            y=-flows_df["depenses"],
                            name="💸 Dépenses",
                            marker_color=color_palette[1],
                            opacity=0.8
                        ))
                        
                        fig_flows.add_trace(go.Scatter(
                            x=flows_df["date"],
                            y=flows_df["net"],
                            mode='lines+markers',
                            name="📊 Flux Net",
                            line=dict(color='orange', width=3)
                        ))
                        
                        fig_flows.update_layout(
                            title="📊 Analyse des Flux Quotidiens",
                            xaxis_title="📅 Date",
                            yaxis_title="💰 Montant (€)",
                            barmode='relative',
                            hovermode="x unified",
                            height=500
                        )
                        st.plotly_chart(fig_flows, width="stretch")

                # Ajout d'une ligne de tendance
                if len(balance_evolution) > 1:
                    # S'assurer que la colonne date est au bon format
                    balance_evolution["date"] = pd.to_datetime(
                        balance_evolution["date"]
                    )

                    # Calcul de la tendance linéaire
                    balance_evolution["days_from_start"] = (
                        balance_evolution["date"] - balance_evolution["date"].min()
                    ).dt.days

                    # Régression linéaire simple
                    x = balance_evolution["days_from_start"]
                    y = balance_evolution["total_euros"]

                    if len(x) > 1:
                        slope = ((x * y).sum() - x.sum() * y.sum() / len(x)) / (
                            (x * x).sum() - x.sum() ** 2 / len(x)
                        )

                        trend_info = (
                            "📈 Tendance positive"
                            if slope > 0
                            else (
                                "📉 Tendance négative"
                                if slope < 0
                                else "➡️ Tendance stable"
                            )
                        )
                        st.info(f"{trend_info} ({slope:.2f} €/jour)")
            else:
                st.info("Pas assez de données pour afficher l'évolution")

        with tab2:
            st.subheader("💰 Répartition des Comptes")

            # Soldes actuels des comptes
            current_balances = compute_balances_at_date(
                st.session_state.ledger, st.session_state.accounts
            )

            # Préparer les données pour le camembert
            account_names = []
            account_values = []

            for acc in st.session_state.accounts:
                balance = current_balances.get(acc["account_id"], 0) / 100.0
                if balance > 0:  # Ne montrer que les comptes avec solde positif
                    account_names.append(acc["name"])
                    account_values.append(balance)

            if account_values:
                fig_pie = px.pie(
                    values=account_values,
                    names=account_names,
                    title="Répartition par compte",
                )
                st.plotly_chart(fig_pie, width="stretch")
            else:
                st.info("Aucun compte avec solde positif")

        with tab3:
            st.subheader("💰 Répartition Détaillée")

            # Calculer les balances actuelles
            current_balances = compute_balances_at_date(
                st.session_state.ledger, st.session_state.accounts
            )

            if current_balances:
                col_pie, col_table = st.columns([1, 1])

                with col_pie:
                    # Graphique en secteurs amélioré
                    account_data = []
                    for acc_id, balance_cents in current_balances.items():
                        if (
                            abs(balance_cents) > 1
                        ):  # Ignorer les comptes avec moins de 1 centime
                            acc_name = next(
                                (
                                    acc["name"]
                                    for acc in st.session_state.accounts
                                    if acc["account_id"] == acc_id
                                ),
                                acc_id,
                            )
                            account_data.append(
                                {
                                    "name": acc_name,
                                    "value": abs(balance_cents / 100.0),
                                    "balance": balance_cents / 100.0,
                                }
                            )

                    if account_data:
                        fig_pie_detailed = px.pie(
                            values=[d["value"] for d in account_data],
                            names=[d["name"] for d in account_data],
                            title="Répartition des soldes (valeurs absolues)",
                            color_discrete_sequence=px.colors.qualitative.Set3,
                        )
                        fig_pie_detailed.update_traces(
                            textposition="inside", textinfo="percent+label"
                        )
                        st.plotly_chart(fig_pie_detailed, width="stretch")

                with col_table:
                    st.write("**Détail complet des soldes:**")
                    total_balance = (
                        sum(
                            balance_cents for balance_cents in current_balances.values()
                        )
                        / 100.0
                    )

                    # Trier par valeur absolue décroissante
                    sorted_accounts = sorted(
                        current_balances.items(), key=lambda x: abs(x[1]), reverse=True
                    )

                    for acc_id, balance_cents in sorted_accounts:
                        acc_name = next(
                            (
                                acc["name"]
                                for acc in st.session_state.accounts
                                if acc["account_id"] == acc_id
                            ),
                            acc_id,
                        )
                        balance_euros = balance_cents / 100.0
                        if abs(balance_euros) > 0.01:
                            color = "green" if balance_euros > 0 else "red"
                            percentage = (
                                abs(balance_euros)
                                / sum(abs(b / 100.0) for b in current_balances.values())
                            ) * 100
                            st.write(
                                f":{color}[{acc_name}: {balance_euros:.2f} € ({percentage:.1f}%)]"
                            )

                    st.write("---")
                    color = "green" if total_balance > 0 else "red"
                    st.write(f"**:{color}[Total: {total_balance:.2f} €]**")
            else:
                st.info("Aucun compte trouvé")

        with tab4:
            st.markdown("### 💎 Simulateur d'Intérêts Interactif")
            
            # Paramètres de simulation dans la sidebar
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🎛️ Paramètres de Simulation")
            
            # Onglets pour différents types de simulation
            sim_tab1, sim_tab2 = st.tabs(["📊 Simulation Personnalisée", "💎 Calculateur NEXO"])
            
            with sim_tab1:
                st.markdown("#### � Simulez votre croissance financière")
                
                # Paramètres de simulation
                col_params1, col_params2 = st.columns(2)
                
                with col_params1:
                    st.markdown("##### 💰 Capital et Apports")
                    initial_capital = st.number_input(
                        "Capital initial (€)", 
                        min_value=0.0, 
                        value=float(sum(balances_by_type["linked"].values()) / 100),
                        step=100.0,
                        help="Montant de départ pour la simulation"
                    )
                    
                    monthly_addition = st.number_input(
                        "Apport mensuel (€)", 
                        min_value=0.0, 
                        value=500.0,
                        step=50.0,
                        help="Montant ajouté chaque mois"
                    )
                    
                    period_months = st.slider(
                        "Période de simulation (mois)", 
                        min_value=1, 
                        max_value=120, 
                        value=24,
                        help="Durée de la simulation en mois"
                    )
                
                with col_params2:
                    st.markdown("##### 📈 Rendements")
                    annual_rate = st.number_input(
                        "Taux annuel (%)", 
                        min_value=0.0, 
                        max_value=50.0, 
                        value=8.5,
                        step=0.1,
                        help="Taux d'intérêt annuel en pourcentage"
                    )
                    
                    nexo_percentage = st.slider(
                        "% Portfolio en NEXO tokens", 
                        min_value=0, 
                        max_value=30, 
                        value=15,
                        help="Pourcentage du portefeuille en tokens NEXO pour les bonus"
                    )
                    
                    nexo_price = st.number_input(
                        "Prix NEXO (€/token)", 
                        min_value=0.1, 
                        value=1.2,
                        step=0.01,
                        help="Prix actuel du token NEXO"
                    )
                
                # Calculer la simulation
                if st.button("🚀 Lancer la Simulation", type="primary", key="run_simulation"):
                    simulation_result = calculate_custom_simulation(
                        initial_capital, 
                        monthly_addition, 
                        annual_rate, 
                        period_months, 
                        nexo_percentage, 
                        nexo_price
                    )
                    
                    # Stocker les résultats dans la session
                    st.session_state.simulation_result = simulation_result
                
                # Afficher les résultats si disponibles
                if hasattr(st.session_state, 'simulation_result') and st.session_state.simulation_result:
                    result = st.session_state.simulation_result
                    summary = result["summary"]
                    
                    st.markdown("---")
                    st.markdown("#### 📊 Résultats de la Simulation")
                    
                    # Cartes résumé
                    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                    
                    with col_r1:
                        st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                       padding: 1rem; border-radius: 8px; color: white; text-align: center;">
                                <h5 style="margin: 0;">💰 Capital Final</h5>
                                <h3 style="margin: 0.5rem 0;">{summary['final_capital']:,.2f} €</h3>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col_r2:
                        st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                       padding: 1rem; border-radius: 8px; color: white; text-align: center;">
                                <h5 style="margin: 0;">� Gains Totaux</h5>
                                <h3 style="margin: 0.5rem 0;">+{summary['total_interest']:,.2f} €</h3>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col_r3:
                        st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                                       padding: 1rem; border-radius: 8px; color: white; text-align: center;">
                                <h5 style="margin: 0;">📊 Taux Effectif</h5>
                                <h3 style="margin: 0.5rem 0;">{summary['effective_rate']:.1f}%</h3>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col_r4:
                        st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #00b894 0%, #00a085 100%); 
                                       padding: 1rem; border-radius: 8px; color: white; text-align: center;">
                                <h5 style="margin: 0;">� Tokens NEXO</h5>
                                <h3 style="margin: 0.5rem 0;">{summary['final_nexo_tokens']:,.0f}</h3>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Graphique d'évolution
                    evolution_df = pd.DataFrame(result["evolution"])
                    
                    fig_evolution = go.Figure()
                    
                    # Capital total
                    fig_evolution.add_trace(go.Scatter(
                        x=evolution_df["month"],
                        y=evolution_df["capital"],
                        mode='lines+markers',
                        name='Capital Total',
                        line=dict(color='#667eea', width=3),
                        fill='tonexty'
                    ))
                    
                    # Capital investi (sans intérêts)
                    capital_invested = evolution_df["month"] * monthly_addition + initial_capital
                    fig_evolution.add_trace(go.Scatter(
                        x=evolution_df["month"],
                        y=capital_invested,
                        mode='lines',
                        name='Capital Investi',
                        line=dict(color='#ff6b6b', width=2, dash='dash')
                    ))
                    
                    fig_evolution.update_layout(
                        title=f"📈 Évolution du capital sur {period_months} mois",
                        xaxis_title="Mois",
                        yaxis_title="Capital (€)",
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_evolution, use_container_width=True)
                    
                    # Tableau détaillé (les 12 derniers mois)
                    if len(evolution_df) > 12:
                        st.markdown("##### 📋 Évolution des 12 derniers mois")
                        display_df = evolution_df.tail(13)  # 12 + le mois initial
                    else:
                        st.markdown("##### 📋 Évolution détaillée")
                        display_df = evolution_df
                    
                    # Formatter le dataframe pour l'affichage
                    formatted_df = display_df.copy()
                    formatted_df["Capital (€)"] = formatted_df["capital"].apply(lambda x: f"{x:,.2f}")
                    formatted_df["Intérêts Mensuels (€)"] = formatted_df["monthly_interest"].apply(lambda x: f"+{x:,.2f}")
                    formatted_df["Tokens NEXO"] = formatted_df["nexo_tokens_needed"].apply(lambda x: f"{x:,.0f}")
                    formatted_df = formatted_df[["month", "Capital (€)", "Intérêts Mensuels (€)", "Tokens NEXO"]].rename(columns={"month": "Mois"})
                    
                    st.dataframe(formatted_df, use_container_width=True, hide_index=True)

            with sim_tab2:
                st.markdown("#### 💎 Calculateur de Tokens NEXO")
                st.caption("Calculez combien de tokens NEXO vous devez détenir pour obtenir les bonus de taux")
                
                # Paramètres du calculateur NEXO
                col_nexo1, col_nexo2 = st.columns(2)
                
                with col_nexo1:
                    st.markdown("##### 💰 Vos Comptes Linked")
                    current_linked = sum(balances_by_type["linked"].values()) / 100
                    
                    custom_linked = st.number_input(
                        "Solde comptes linked (€)", 
                        min_value=0.0, 
                        value=current_linked,
                        step=100.0,
                        help="Total des comptes linked sur Nexo"
                    )
                    
                    current_nexo_price = st.number_input(
                        "Prix NEXO actuel (€)", 
                        min_value=0.01, 
                        value=1.2,
                        step=0.01,
                        help="Prix actuel du token NEXO"
                    )
                
                with col_nexo2:
                    st.markdown("##### 🎯 Objectifs NEXO")
                    target_percentages = st.multiselect(
                        "Niveaux cibles (%)",
                        [10, 12, 15, 18, 20, 25],
                        default=[10, 15, 20],
                        help="Pourcentages de NEXO tokens à calculer"
                    )
                
                if custom_linked > 0 and current_nexo_price > 0 and target_percentages:
                    st.markdown("---")
                    st.markdown("#### 📊 Analyse des Besoins NEXO")
                    
                    # Calculer pour chaque pourcentage cible
                    nexo_calculations = []
                    
                    for target_pct in sorted(target_percentages):
                        calc = calculate_nexo_requirements(custom_linked, target_pct, current_nexo_price)
                        if "error" not in calc:
                            nexo_calculations.append({
                                "% Cible": f"{target_pct}%",
                                "Tokens NEXO": f"{calc['tokens_needed']:,.0f}",
                                "Valeur (€)": f"{calc['target_nexo_value_eur']:,.2f}",
                                "Bonus APY": f"+{calc['bonus_rate']:.1f}%",
                                "Coût": f"{calc['investment_cost']:,.2f} €"
                            })
                    
                    if nexo_calculations:
                        # Tableau des besoins
                        df_nexo_calc = pd.DataFrame(nexo_calculations)
                        st.dataframe(df_nexo_calc, use_container_width=True, hide_index=True)
                        
                        # Graphique comparatif
                        fig_nexo_req = go.Figure()
                        
                        percentages = [int(calc["% Cible"].rstrip('%')) for calc in nexo_calculations]
                        tokens_needed = [float(calc["Tokens NEXO"].replace(',', '')) for calc in nexo_calculations]
                        costs = [float(calc["Coût"].replace(',', '').replace(' €', '')) for calc in nexo_calculations]
                        
                        # Barres pour les tokens
                        fig_nexo_req.add_trace(go.Bar(
                            x=[f"{p}%" for p in percentages],
                            y=tokens_needed,
                            name="Tokens NEXO nécessaires",
                            marker_color='#667eea',
                            text=[f"{t:,.0f}" for t in tokens_needed],
                            textposition='auto',
                            yaxis='y'
                        ))
                        
                        # Ligne pour les coûts
                        fig_nexo_req.add_trace(go.Scatter(
                            x=[f"{p}%" for p in percentages],
                            y=costs,
                            mode='lines+markers',
                            name="Coût d'investissement (€)",
                            line=dict(color='#f093fb', width=3),
                            marker=dict(size=8),
                            yaxis='y2'
                        ))
                        
                        fig_nexo_req.update_layout(
                            title=f"💎 Besoins en Tokens NEXO pour {custom_linked:,.0f}€",
                            xaxis_title="Pourcentage NEXO cible",
                            yaxis=dict(title="Nombre de tokens", side='left'),
                            yaxis2=dict(title="Coût (€)", side='right', overlaying='y'),
                            height=400,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_nexo_req, use_container_width=True)
                        
                        # Recommandations automatiques
                        st.markdown("#### 💡 Recommandations")
                        
                        # Trouver le meilleur rapport qualité/prix
                        best_value_idx = 0
                        if len(nexo_calculations) > 1:
                            # Privilégier 15% comme bon compromis
                            for i, calc in enumerate(nexo_calculations):
                                if calc["% Cible"] == "15%":
                                    best_value_idx = i
                                    break
                        
                        best_calc = nexo_calculations[best_value_idx] if nexo_calculations else None
                        
                        if best_calc:
                            col_rec1, col_rec2 = st.columns(2)
                            
                            with col_rec1:
                                st.success(f"""
                                **🎯 Recommandation Optimale: {best_calc['% Cible']}**
                                
                                - 🪙 Tokens nécessaires: **{best_calc['Tokens NEXO']}**
                                - 💰 Investissement: **{best_calc['Coût']}**
                                - 📈 Bonus obtenu: **{best_calc['Bonus APY']}**
                                - 🎁 Excellent rapport rendement/risque
                                """)
                            
                            with col_rec2:
                                # Calcul du ROI annuel du bonus
                                bonus_rate = float(best_calc['Bonus APY'].replace('+', '').replace('%', ''))
                                annual_bonus = custom_linked * (bonus_rate / 100)
                                investment_cost = float(best_calc['Coût'].replace(',', '').replace(' €', ''))
                                
                                st.info(f"""
                                **📊 Analyse du Retour sur Investissement**
                                
                                - 💵 Bonus annuel: **+{annual_bonus:.2f}€**
                                - 🔄 ROI du bonus: **{(annual_bonus/investment_cost*100):.1f}%**
                                - ⏱️ Amortissement: **{(investment_cost/annual_bonus):.1f} ans**
                                - ✨ + Potentiel d'appréciation du token NEXO
                                """)
                        
                        # Zone d'alerte pour les gros investissements
                        max_cost = max(costs) if costs else 0
                        if max_cost > custom_linked * 0.3:  # Plus de 30% du portefeuille
                            st.warning(f"""
                            ⚠️ **Attention:** Les niveaux élevés de NEXO (20%+) nécessitent un investissement important 
                            ({max_cost:,.0f}€ pour 20%). Considérez une approche progressive :
                            
                            1. Commencer par 10-15% de NEXO
                            2. Réinvestir les gains en tokens NEXO
                            3. Augmenter progressivement selon les performances
                            """)

        with tab5:
            st.subheader("🔮 Prédictions et Analyse des Flux")

            if show_predictions:
                # Analyse des flux récents
                flows_df = get_daily_flow_analysis(st.session_state.ledger, period_days)

                if not flows_df.empty:
                    col_pred1, col_pred2 = st.columns([2, 1])

                    with col_pred1:
                        # Graphique des flux avec moyennes mobiles
                        fig_flows_pred = go.Figure()

                        fig_flows_pred.add_trace(
                            go.Bar(
                                x=flows_df["date"],
                                y=flows_df["recettes"],
                                name="Recettes",
                                marker_color="green",
                                opacity=0.7,
                            )
                        )

                        fig_flows_pred.add_trace(
                            go.Bar(
                                x=flows_df["date"],
                                y=-flows_df["depenses"],
                                name="Dépenses",
                                marker_color="red",
                                opacity=0.7,
                            )
                        )

                        # Moyennes mobiles si on a assez de données
                        if len(flows_df) > 7:
                            flows_df["recettes_ma"] = (
                                flows_df["recettes"].rolling(7, min_periods=1).mean()
                            )
                            flows_df["depenses_ma"] = (
                                flows_df["depenses"].rolling(7, min_periods=1).mean()
                            )

                            fig_flows_pred.add_trace(
                                go.Scatter(
                                    x=flows_df["date"],
                                    y=flows_df["recettes_ma"],
                                    mode="lines",
                                    name="Trend Recettes (7j)",
                                    line=dict(color="darkgreen", width=2),
                                )
                            )

                            fig_flows_pred.add_trace(
                                go.Scatter(
                                    x=flows_df["date"],
                                    y=-flows_df["depenses_ma"],
                                    mode="lines",
                                    name="Trend Dépenses (7j)",
                                    line=dict(color="darkred", width=2),
                                )
                            )

                        fig_flows_pred.update_layout(
                            title=f"Analyse des flux sur {period_days} jours",
                            xaxis_title="Date",
                            yaxis_title="Montant (€)",
                            barmode="relative",
                            showlegend=True,
                        )

                        st.plotly_chart(fig_flows_pred, width="stretch")

                    with col_pred2:
                        # Métriques de prédiction
                        total_recettes = flows_df["recettes"].sum()
                        total_depenses = flows_df["depenses"].sum()
                        avg_daily_net = (total_recettes - total_depenses) / len(
                            flows_df
                        )

                        st.metric("💹 Flux net moyen/jour", f"{avg_daily_net:.2f} €")

                        # Prédictions simples
                        pred_7j = avg_daily_net * 7
                        pred_30j = avg_daily_net * 30
                        pred_horizon = avg_daily_net * prediction_days

                        st.metric("📈 Prédiction 7j", f"{pred_7j:+.2f} €")
                        st.metric("📅 Prédiction 30j", f"{pred_30j:+.2f} €")
                        st.metric(
                            f"🔮 Prédiction {prediction_days}j",
                            f"{pred_horizon:+.2f} €",
                        )

                        # Volatilité
                        if len(flows_df) > 1:
                            daily_net = flows_df["recettes"] - flows_df["depenses"]
                            volatility = daily_net.std()
                            st.metric(
                                "📊 Volatilité quotidienne", f"{volatility:.2f} €"
                            )

                        # Conseils basés sur les tendances
                        st.subheader("💡 Insights")
                        if avg_daily_net > 0:
                            st.success(
                                f"🟢 Tendance positive: +{avg_daily_net:.2f}€/jour en moyenne"
                            )
                        elif avg_daily_net < 0:
                            st.warning(
                                f"🟡 Tendance négative: {avg_daily_net:.2f}€/jour en moyenne"
                            )
                        else:
                            st.info("⚪ Tendance neutre")

                        # Prédiction de date d'épuisement/enrichissement
                        current_balance = kpis.get("solde_actuel", 0)
                        if avg_daily_net < 0 and current_balance > 0:
                            days_to_zero = current_balance / abs(avg_daily_net)
                            if days_to_zero < 90:
                                st.error(
                                    f"⚠️ Solde épuisé dans ~{days_to_zero:.0f} jours au rythme actuel"
                                )
                        elif avg_daily_net > 0:
                            days_to_double = (
                                current_balance / avg_daily_net
                                if current_balance > 0
                                else float("inf")
                            )
                            if days_to_double < 365:
                                st.success(
                                    f"📈 Solde doublé dans ~{days_to_double:.0f} jours au rythme actuel"
                                )

                else:
                    st.info("Pas assez de données pour les prédictions")
            else:
                st.info(
                    "Activez les prédictions dans la sidebar pour voir cette section"
                )

        with tab6:
            st.markdown("#### 🎯 Insights & Recommandations")
            
            # Analyse intelligente basée sur les données
            col_insights, col_reco = st.columns([1, 1])
            
            with col_insights:
                st.markdown("##### 🔍 Analyse Automatique")
                
                # Calculs pour les insights
                linked_total = sum(balances_by_type["linked"].values()) / 100
                unlinked_total = sum(balances_by_type["unlinked"].values()) / 100
                total_net_worth = linked_total + unlinked_total
                
                # Ratios d'analyse
                if total_net_worth > 0:
                    linked_ratio = (linked_total / total_net_worth) * 100
                    unlinked_ratio = (unlinked_total / total_net_worth) * 100
                    
                    # Insights automatiques
                    insights = []
                    
                    if linked_ratio > 80:
                        insights.append("⚠️ **Liquidité élevée**: Plus de 80% de vos fonds sont en comptes linked (opérationnels)")
                    elif linked_ratio < 20:
                        insights.append("💰 **Épargne importante**: Plus de 80% de vos fonds sont en comptes unlinked")
                    
                    if kpis.get('tendance_7j', 0) > 0:
                        insights.append(f"📈 **Tendance positive**: +{kpis.get('tendance_7j', 0):.2f}€ sur 7 jours")
                    elif kpis.get('tendance_7j', 0) < -50:
                        insights.append(f"📉 **Attention**: Tendance négative de {kpis.get('tendance_7j', 0):.2f}€ sur 7j")
                    
                    avg_expense = kpis.get('depense_moy_jour', 0)
                    if avg_expense > 0 and linked_total > 0:
                        autonomy_days = linked_total / avg_expense
                        if autonomy_days < 30:
                            insights.append(f"🚨 **Autonomie limitée**: Seulement {autonomy_days:.0f} jours d'autonomie")
                        elif autonomy_days > 365:
                            insights.append(f"✅ **Excellente autonomie**: Plus d'1 an d'autonomie financière")
                    
                    # Analyse de la volatilité
                    volatility = abs(kpis.get('tendance_7j', 0)) / 7
                    if volatility > 20:
                        insights.append("📊 **Volatilité élevée**: Variations importantes dans vos finances")
                    elif volatility < 5:
                        insights.append("📊 **Finances stables**: Faible volatilité détectée")
                    
                    for insight in insights:
                        st.markdown(f"- {insight}")
                        
                    if not insights:
                        st.info("💡 Pas d'insights particuliers détectés. Continuez à alimenter vos données!")
                
                # Statistiques avancées
                st.markdown("##### 📊 Statistiques Avancées")
                
                stats_data = {
                    "Métrique": [
                        "Ratio Linked/Total",
                        "Autonomie Financière", 
                        "Croissance Mensuelle",
                        "Score de Diversification"
                    ],
                    "Valeur": [
                        f"{linked_ratio:.1f}%",
                        f"{linked_total/avg_expense:.0f} jours" if avg_expense > 0 else "∞",
                        f"{kpis.get('tendance_7j', 0)*4.3:+.0f} €/mois",
                        f"{min(len([acc for acc in st.session_state.accounts if not acc.get('is_unlinked', False)]), 5)}/5"
                    ]
                }
                
                st.dataframe(pd.DataFrame(stats_data), width='stretch', hide_index=True)
            
            with col_reco:
                st.markdown("##### 💡 Recommandations Personnalisées")
                
                recommendations = []
                
                # Recommandations basées sur les ratios
                if linked_ratio > 85:
                    recommendations.append({
                        "type": "💰 Épargne",
                        "titre": "Diversifier votre épargne",
                        "description": "Considérez créer des comptes unlinked pour séparer épargne et liquidités",
                        "priorité": "Moyenne"
                    })
                
                if unlinked_total < 1000 and linked_total > 2000:
                    recommendations.append({
                        "type": "🎯 Planification",
                        "titre": "Constituer une épargne de précaution",
                        "description": "Transférez une partie vers des comptes unlinked pour l'épargne",
                        "priorité": "Haute"
                    })
                
                if avg_expense > 0 and linked_total / avg_expense < 60:
                    recommendations.append({
                        "type": "🚨 Urgent",
                        "titre": "Renforcer la trésorerie",
                        "description": "Moins de 2 mois d'autonomie. Augmentez vos comptes linked",
                        "priorité": "Critique"
                    })
                
                if kpis.get('tendance_7j', 0) < -100:
                    recommendations.append({
                        "type": "📉 Contrôle",
                        "titre": "Analyser les dépenses",
                        "description": "Tendance négative importante. Révisez votre budget",
                        "priorité": "Haute"
                    })
                
                if len(st.session_state.accounts) < 3:
                    recommendations.append({
                        "type": "🏦 Organisation",
                        "titre": "Diversifier vos comptes",
                        "description": "Créez des comptes spécialisés (épargne, projets, etc.)",
                        "priorité": "Basse"
                    })
                
                # Affichage des recommandations
                for i, reco in enumerate(recommendations):
                    priority_color = {
                        "Critique": "#ff4444",
                        "Haute": "#ff8800", 
                        "Moyenne": "#ffbb00",
                        "Basse": "#00bb00"
                    }.get(reco["priorité"], "#666666")
                    
                    st.markdown(f"""
                        <div style="border-left: 4px solid {priority_color}; 
                                   padding: 1rem; margin: 1rem 0; 
                                   background: #f8f9fa; border-radius: 0 8px 8px 0;">
                            <h4 style="margin: 0; color: {priority_color};">{reco['type']}</h4>
                            <h5 style="margin: 0.5rem 0;">{reco['titre']}</h5>
                            <p style="margin: 0; color: #666;">{reco['description']}</p>
                            <small style="color: {priority_color};">Priorité: {reco['priorité']}</small>
                        </div>
                    """, unsafe_allow_html=True)
                
                if not recommendations:
                    st.success("✅ Vos finances semblent bien organisées ! Aucune recommandation urgente.")
            
            # Section objectifs et planification
            st.markdown("---")
            st.markdown("##### 🎯 Planification d'Objectifs")
            
            col_obj1, col_obj2 = st.columns(2)
            
            with col_obj1:
                st.markdown("**💰 Simulateur d'Épargne**")
                target_amount = st.number_input("Objectif d'épargne (€)", min_value=0.0, value=5000.0, step=100.0)
                monthly_savings = st.number_input("Épargne mensuelle (€)", min_value=0.0, value=200.0, step=10.0)
                
                if monthly_savings > 0:
                    months_needed = target_amount / monthly_savings
                    st.info(f"⏱️ Temps nécessaire: **{months_needed:.1f} mois** ({months_needed/12:.1f} ans)")
                    
            with col_obj2:
                st.markdown("**📊 Projection Patrimoine**")
                if kpis.get('tendance_7j', 0) != 0:
                    monthly_trend = kpis.get('tendance_7j', 0) * 4.3  # 7j * 4.3 ≈ 30j
                    current_net_worth = total_net_worth
                    
                    projections = []
                    for months in [3, 6, 12]:
                        future_value = current_net_worth + (monthly_trend * months)
                        projections.append(f"**{months}M**: {future_value:,.0f}€")
                    
                    st.write("Projections basées sur la tendance:")
                    for proj in projections:
                        st.write(f"- {proj}")
                else:
                    st.info("Pas assez de données pour les projections")
