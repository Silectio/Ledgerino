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


# UI
st.title("💰 Ledger (simple)")

if not st.session_state.pseudo:
    st.warning(
        "Saisissez un pseudo dans la barre latérale puis cliquez 'Charger cet utilisateur'."
    )
elif page == "Ledger":
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Comptes et soldes")
        balances_by_type = compute_balances_by_type()

        # Préparer les données pour l'affichage
        rows_acc = []
        for acc in st.session_state.accounts:
            status_icon = "🔗" if not acc.get("is_unlinked", False) else "📎"
            status_text = "Linked" if not acc.get("is_unlinked", False) else "Unlinked"
            balance_eur = (
                balances_by_type[
                    "linked" if not acc.get("is_unlinked", False) else "unlinked"
                ].get(acc["account_id"], 0)
                / 100
            )

            rows_acc.append(
                {
                    "Nom": f"{status_icon} {acc['name']}",
                    "Type": status_text,
                    "Solde (€)": f"{balance_eur:.2f}",
                }
            )

        # Calculer les totaux
        linked_total = sum(balances_by_type["linked"].values()) / 100
        unlinked_total = sum(balances_by_type["unlinked"].values()) / 100
        grand_total = linked_total + unlinked_total

        # Ajouter les lignes de totaux
        rows_acc.append(
            {"Nom": "━━━━━━━━━━━━━━", "Type": "━━━━━━━", "Solde (€)": "━━━━━━━━"}
        )
        rows_acc.append(
            {
                "Nom": "🔗 Total Linked",
                "Type": "Subtotal",
                "Solde (€)": f"{linked_total:.2f}",
            }
        )
        rows_acc.append(
            {
                "Nom": "📎 Total Unlinked",
                "Type": "Subtotal",
                "Solde (€)": f"{unlinked_total:.2f}",
            }
        )
        rows_acc.append(
            {
                "Nom": "💰 TOTAL GÉNÉRAL",
                "Type": "Total",
                "Solde (€)": f"{grand_total:.2f}",
            }
        )

        df_acc = pd.DataFrame(rows_acc)
        st.dataframe(df_acc, width="stretch")

        # Bouton pour supprimer la dernière opération du ledger
        if st.session_state.ledger:
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
                meta = " | ".join(
                    p
                    for p in [
                        t,
                        f"{amt:.2f}€",
                        f"depuis {src}" if src else "",
                        f"vers {dest}" if dest else "",
                        f"compte {accn}" if accn else "",
                    ]
                    if p
                )
                st.caption(f"Dernière: {last.get('ts','')} · {meta}")
            except Exception:
                pass
            if st.button("Supprimer la dernière opération", key="delete_last_op"):
                if delete_last_ledger_entry(st.session_state.pseudo):
                    # Recharger le ledger depuis la DB
                    st.session_state.ledger = load_user_ledger(st.session_state.pseudo)
                    st.success("Dernière opération supprimée")
                    st.rerun()
                else:
                    st.warning("Aucune opération à supprimer")

        # Exécution rapide des règles (boutons)
        if st.session_state.rules:
            st.subheader("Règles disponibles")
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

    with col_right:
        st.subheader("Ajouter une opération manuelle")
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

    st.subheader("Journal des opérations")
    if st.session_state.ledger:
        acc_id_to_name = {
            acc["account_id"]: acc["name"] for acc in st.session_state.accounts
        }
        rows = []
        for op in reversed(st.session_state.ledger):
            t = op.get("type")
            amt = op.get("amount_cents", 0) / 100
            rows.append(
                {
                    "Date (UTC)": op.get("ts"),
                    "Type": t,
                    "Depuis": acc_id_to_name.get(op.get("src_account_id", ""), ""),
                    "Vers": acc_id_to_name.get(op.get("dest_account_id", ""), ""),
                    "Compte": acc_id_to_name.get(op.get("account_id", ""), ""),
                    "Montant (€)": f"{amt:.2f}",
                    "Note": op.get("note", ""),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch")
    else:
        st.info("Aucune opération pour l'instant")

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
    st.header("📊 Dashboard & Analyses Avancées")

    if not st.session_state.pseudo:
        st.warning("Sélectionnez un utilisateur pour voir le dashboard.")
    elif not st.session_state.ledger:
        st.info(
            "Aucune donnée disponible. Ajoutez des opérations pour voir les analyses."
        )
    else:
        # Sidebar avec filtres
        st.sidebar.header("🔧 Filtres & Options")

        # Créer un dictionnaire des comptes pour faciliter l'utilisation
        accounts_dict = {acc["account_id"]: acc for acc in st.session_state.accounts}

        # Filtre par comptes
        account_options = ["Tous les comptes"] + [
            f"{acc['name']} ({acc['account_id']})" for acc in st.session_state.accounts
        ]
        selected_accounts_display = st.sidebar.multiselect(
            "Comptes à analyser", account_options, default=["Tous les comptes"]
        )

        # Convertir la sélection en IDs de comptes
        if "Tous les comptes" in selected_accounts_display:
            selected_account_ids = list(accounts_dict.keys())
        else:
            selected_account_ids = []
            for selection in selected_accounts_display:
                if selection != "Tous les comptes":
                    # Extraire l'ID du compte depuis "(ID)"
                    acc_id = selection.split("(")[-1].strip(")")
                    selected_account_ids.append(acc_id)

        # Période d'analyse
        period_days = st.sidebar.selectbox(
            "Période d'analyse",
            [7, 15, 30, 60, 90, 180],
            index=2,  # 30 jours par défaut
        )

        # Options de prédiction
        st.sidebar.subheader("🔮 Prédictions")
        enable_predictions = st.sidebar.checkbox("Activer les prédictions", value=True)
        prediction_days = (
            st.sidebar.slider("Horizon de prédiction (jours)", 7, 90, 30)
            if enable_predictions
            else 0
        )

        # Calcul des KPIs
        kpis = calculate_kpis(st.session_state.ledger, st.session_state.accounts)

        # Section KPIs en haut
        st.subheader("📈 Indicateurs Clés")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="💰 Solde Actuel",
                value=f"{kpis.get('solde_actuel', 0):.2f} €",
                delta=f"{kpis.get('tendance_7j', 0):+.2f} € (7j)",
            )

        with col2:
            st.metric(label="📊 Moyenne 7j", value=f"{kpis.get('moyenne_7j', 0):.2f} €")

        with col3:
            st.metric(
                label="📅 Moyenne 30j", value=f"{kpis.get('moyenne_30j', 0):.2f} €"
            )

        with col4:
            st.metric(
                label="💸 Dépense/jour",
                value=f"{kpis.get('depense_moy_jour', 0):.2f} €",
            )

        st.markdown("---")

        # Onglets pour organiser les différentes vues
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📈 Évolution Globale",
                "📊 Par Compte",
                "💰 Répartition",
                "🔮 Prédictions",
            ]
        )

        with tab1:
            st.subheader("📈 Évolution du Solde Total")

            # Récupération des données d'évolution
            balance_evolution = get_balance_evolution(
                st.session_state.ledger, st.session_state.accounts, period_days
            )

            if not balance_evolution.empty:
                # Graphique principal de l'évolution
                fig_balance = px.line(
                    balance_evolution,
                    x="date",
                    y="total_euros",
                    title=f"Évolution du solde total sur {period_days} jours",
                    labels={"total_euros": "Solde (€)", "date": "Date"},
                )
                fig_balance.update_layout(
                    xaxis_title="Date", yaxis_title="Solde (€)", hovermode="x unified"
                )
                st.plotly_chart(fig_balance, width="stretch")

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
            st.subheader("🔮 Prédictions et Analyse des Flux")

            if enable_predictions:
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
