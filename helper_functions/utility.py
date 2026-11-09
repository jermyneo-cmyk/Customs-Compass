import streamlit as st
import hmac

def _users():
    """Read the two users from secrets. Falls back to demo values locally
    if secrets are not set, so the app still runs during development."""
    try:
        return dict(st.secrets["users"])
    except Exception:
        return {"trader1": "password1", "trader2": "password2"}

def check_password():
    """Show a login form. Return True once a valid user/password is entered."""
    if st.session_state.get("authenticated"):
        return True

    st.subheader("Customs Compass — Sign in")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        users = _users()
        stored = users.get(username)
        # hmac.compare_digest avoids leaking timing information
        if stored and hmac.compare_digest(password, stored):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Incorrect username or password.")

    return False