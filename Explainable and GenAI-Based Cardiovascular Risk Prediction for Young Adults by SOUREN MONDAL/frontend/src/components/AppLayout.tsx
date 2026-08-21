import {
  Activity,
  ClipboardPlus,
  HeartPulse,
  History,
  LayoutDashboard,
  LogOut,
  Menu,
  X,
} from "lucide-react";
import { signOut } from "firebase/auth";
import { useState, type ReactNode } from "react";
import {
  NavLink,
  useNavigate,
} from "react-router-dom";

import { useAuth } from "../context/AuthContext";
import { auth } from "../lib/firebase";

type AppLayoutProps = {
  children: ReactNode;
};

export function AppLayout({
  children,
}: AppLayoutProps) {
  const navigate = useNavigate();
  const { user } = useAuth();

  const [menuOpen, setMenuOpen] = useState(false);

  async function handleLogout() {
    await signOut(auth);
    navigate("/login");
  }

  const navigationItems = [
    {
      label: "Dashboard",
      path: "/dashboard",
      icon: LayoutDashboard,
    },
    {
      label: "New assessment",
      path: "/assessment",
      icon: ClipboardPlus,
    },
    {
      label: "Prediction history",
      path: "/history",
      icon: History,
    },
  ];

  return (
    <div className="app-shell">
      <header className="app-header">
        <NavLink
          to="/dashboard"
          className="brand"
          onClick={() => setMenuOpen(false)}
        >
          <span className="brand-icon">
            <HeartPulse size={25} />
          </span>

          <span>
            <strong>CardioInsight</strong>
            <small>Heart Risk X-GenAI</small>
          </span>
        </NavLink>

        <button
          type="button"
          className="mobile-menu-button"
          onClick={() =>
            setMenuOpen((current) => !current)
          }
          aria-label="Toggle navigation"
        >
          {menuOpen ? <X /> : <Menu />}
        </button>

        <nav
          className={
            menuOpen
              ? "app-navigation app-navigation-open"
              : "app-navigation"
          }
        >
          {navigationItems.map((item) => {
            const Icon = item.icon;

            return (
              <NavLink
                key={item.path}
                to={item.path}
                onClick={() => setMenuOpen(false)}
                className={({ isActive }) =>
                  isActive
                    ? "navigation-link navigation-link-active"
                    : "navigation-link"
                }
              >
                <Icon size={18} />
                {item.label}
              </NavLink>
            );
          })}
        </nav>

        <div className="header-user">
          <div className="user-avatar">
            {(user?.displayName ||
              user?.email ||
              "U")
              .charAt(0)
              .toUpperCase()}
          </div>

          <div className="user-summary">
            <strong>
              {user?.displayName || "User"}
            </strong>
            <small>{user?.email}</small>
          </div>

          <button
            type="button"
            className="logout-icon-button"
            onClick={handleLogout}
            title="Log out"
          >
            <LogOut size={19} />
          </button>
        </div>
      </header>

      <div className="application-background">
        <div className="background-orb orb-one" />
        <div className="background-orb orb-two" />
      </div>

      <main className="application-content">
        {children}
      </main>

      <footer className="app-footer">
        <Activity size={16} />

        <span>
          Research-based cardiovascular risk screening
          with explainable AI
        </span>
      </footer>
    </div>
  );
}