"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  auth.py — Supabase Authentication Module                                    ║
║  Handles: signup, login, logout, session state                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
from db import get_supabase_client


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def init_session() -> None:
    """Initialise all authentication keys in session_state on first run."""
    defaults = {
        "user":          None,   # Supabase User object (or None)
        "access_token":  None,   # JWT access token string
        "auth_mode":     "login" # "login" | "signup"
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def is_authenticated() -> bool:
    """Return True if a user is currently logged in."""
    return st.session_state.get("user") is not None


def current_user():
    """Return the current Supabase User object, or None."""
    return st.session_state.get("user")


def current_user_id() -> str | None:
    """Return the current user's UUID string, or None."""
    user = current_user()
    return str(user.id) if user else None


def current_user_email() -> str | None:
    """Return the current user's email, or None."""
    user = current_user()
    return user.email if user else None


# ─────────────────────────────────────────────────────────────────────────────
# CORE AUTH OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def sign_up(email: str, password: str) -> tuple[bool, str]:
    """
    Register a new user with Supabase Auth.

    Returns:
        (success: bool, message: str)
    """
    email    = email.strip().lower()
    password = password.strip()

    # Basic client-side validation
    if not email or "@" not in email:
        return False, "Please enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    try:
        client   = get_supabase_client()
        response = client.auth.sign_up(
            {"email": email, "password": password}
        )

        # Supabase returns user=None when email already exists
        if response.user is None:
            return False, "This email is already registered. Please log in."

        # Store session immediately if email confirmation is disabled
        if response.session:
            st.session_state["user"]         = response.user
            st.session_state["access_token"] = response.session.access_token
            return True, "Account created! Welcome to HealthAI Explainer."

        # Email confirmation required (Supabase default for new projects)
        return True, "Account created! Check your email to confirm your account, then log in."

    except Exception as e:
        err = str(e).lower()
        if "already registered" in err or "duplicate" in err:
            return False, "This email is already registered. Please log in."
        if "password" in err:
            return False, "Password is too weak. Use at least 6 characters."
        return False, f"Signup failed: {e}"


def sign_in(email: str, password: str) -> tuple[bool, str]:
    """
    Authenticate an existing user with Supabase Auth.

    Returns:
        (success: bool, message: str)
    """
    email    = email.strip().lower()
    password = password.strip()

    if not email or not password:
        return False, "Please enter both email and password."

    try:
        client   = get_supabase_client()
        response = client.auth.sign_in_with_password(
            {"email": email, "password": password}
        )

        st.session_state["user"]         = response.user
        st.session_state["access_token"] = response.session.access_token
        return True, f"Welcome back, {response.user.email}!"

    except Exception as e:
        err = str(e).lower()
        if "invalid login" in err or "invalid credentials" in err or "email not confirmed" in err:
            return False, "Invalid email or password. Please try again."
        if "email not confirmed" in err:
            return False, "Please confirm your email before logging in."
        return False, f"Login failed: {e}"


def sign_out() -> None:
    """Log the current user out and clear session state."""
    try:
        client = get_supabase_client()
        client.auth.sign_out()
    except Exception:
        pass  # Always clear local state even if API call fails
    finally:
        st.session_state["user"]         = None
        st.session_state["access_token"] = None


# ─────────────────────────────────────────────────────────────────────────────
# AUTH UI — Login / Signup forms rendered in the main area
# ─────────────────────────────────────────────────────────────────────────────

def render_auth_page() -> None:
    """
    Render the full authentication page (login + signup toggle).
    Called from app.py when the user is not authenticated.
    """
    # Centre the form with column layout
    _, centre, _ = st.columns([1, 1.6, 1])

    with centre:
        # ── Logo / branding ───────────────────────────────────────────────────
        st.markdown("""
        <div style='text-align:center; padding: 32px 0 8px 0;'>
            <div style='font-size:3.5rem;'></div>
            <div style='font-size:1.6rem; font-weight:700; color:#0f172a;
                        letter-spacing:-0.5px; margin-top:8px;'>
                Multi Disease Prediction Using SHAP
            </div>
            <div style='font-size:0.9rem; color:#64748b; margin-top:4px;'>
                Explainable Disease Prediction — Powered by ML + SHAP
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Mode toggle ───────────────────────────────────────────────────────
        mode_col1, mode_col2 = st.columns(2)
        with mode_col1:
            if st.button(
                "Log In",
                use_container_width=True,
                type="primary" if st.session_state["auth_mode"] == "login" else "secondary",
                key="btn_mode_login",
            ):
                st.session_state["auth_mode"] = "login"
                st.rerun()
        with mode_col2:
            if st.button(
                "Sign Up",
                use_container_width=True,
                type="primary" if st.session_state["auth_mode"] == "signup" else "secondary",
                key="btn_mode_signup",
            ):
                st.session_state["auth_mode"] = "signup"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Form card ─────────────────────────────────────────────────────────
        with st.container(border=True):

            if st.session_state["auth_mode"] == "login":
                # ── LOGIN ─────────────────────────────────────────────────────
                st.markdown("#### Log in to your account")
                email    = st.text_input("Email address", placeholder="you@example.com",
                                          key="login_email")
                password = st.text_input("Password", type="password",
                                          placeholder="Your password",
                                          key="login_password")

                st.markdown("<br>", unsafe_allow_html=True)

                if st.button("Log In →", use_container_width=True,
                              type="primary", key="btn_login"):
                    if not email or not password:
                        st.error("Please fill in both fields.")
                    else:
                        with st.spinner("Logging in..."):
                            ok, msg = sign_in(email, password)
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

                st.markdown("""
                <div style='text-align:center; font-size:0.82rem;
                            color:#64748b; margin-top:12px;'>
                    Don't have an account?
                    Switch to <b>Sign Up</b> above.
                </div>""", unsafe_allow_html=True)

            else:
                # ── SIGNUP ────────────────────────────────────────────────────
                st.markdown("#### Create your account")
                email    = st.text_input("Email address", placeholder="you@example.com",
                                          key="signup_email")
                password = st.text_input("Password", type="password",
                                          placeholder="At least 6 characters",
                                          key="signup_password")
                confirm  = st.text_input("Confirm password", type="password",
                                          placeholder="Repeat your password",
                                          key="signup_confirm")

                st.markdown("<br>", unsafe_allow_html=True)

                if st.button("Create Account →", use_container_width=True,
                              type="primary", key="btn_signup"):
                    if not email or not password or not confirm:
                        st.error("Please fill in all fields.")
                    elif password != confirm:
                        st.error("Passwords do not match.")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters.")
                    else:
                        with st.spinner("Creating account..."):
                            ok, msg = sign_up(email, password)
                        if ok:
                            st.success(msg)
                            if is_authenticated():
                                st.rerun()
                        else:
                            st.error(msg)

                st.markdown("""
                <div style='text-align:center; font-size:0.82rem;
                            color:#64748b; margin-top:12px;'>
                    Already have an account?
                    Switch to <b>Log In</b> above.
                </div>""", unsafe_allow_html=True)

        # ── Disclaimer below form ─────────────────────────────────────────────
        st.markdown("""
        <div style='text-align:center; font-size:0.78rem; color:#94a3b8;
                    margin-top:20px; padding:0 8px;'>
            Your data is stored securely via Supabase.<br>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR USER WIDGET
# Called from app.py inside the `with st.sidebar:` block.
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar_user_widget() -> None:
    """Render logged-in user email + logout button in the sidebar."""
    email = current_user_email() or "Unknown"
    # Truncate long emails for display
    display = email if len(email) <= 24 else email[:21] + "..."

    st.markdown(f"""
    <div style='background:#1e293b; border-radius:10px; padding:12px 14px;
                margin: 8px 0 4px 0;'>
        <div style='font-size:0.7rem; color:#64748b; margin-bottom:3px;'>
            Logged in as
        </div>
        <div style='font-size:0.85rem; color:#e2e8f0; font-weight:500;
                    word-break:break-all;'>
            {display}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Log Out", use_container_width=True, key="btn_logout"):
        sign_out()
        st.rerun()
