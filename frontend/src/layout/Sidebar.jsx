import { Box, Typography, Button } from "@mui/material";
import { NavLink, useNavigate } from "react-router-dom";
import DashboardOutlinedIcon from "@mui/icons-material/DashboardOutlined";
import AddCircleOutlineOutlinedIcon from "@mui/icons-material/AddCircleOutlineOutlined";
import HistoryOutlinedIcon from "@mui/icons-material/HistoryOutlined";
import SettingsOutlinedIcon from "@mui/icons-material/SettingsOutlined";
import LogoutRoundedIcon from "@mui/icons-material/LogoutRounded";
import { useAuth } from "../hooks/useAuth";

import krishnaGlow from "../assets/images/krishna-glow.png";
import featherGlow from "../assets/images/feather-glow.png"; // ✅ premium feather

const navItems = [
  { label: "Dashboard", icon: <DashboardOutlinedIcon />, to: "/dashboard" },
  {
    label: "Submit Thought",
    icon: <AddCircleOutlineOutlinedIcon />,
    to: "/submit",
  },
  { label: "History", icon: <HistoryOutlinedIcon />, to: "/history" },
  { label: "Settings", icon: <SettingsOutlinedIcon />, to: "/settings" },
];

export default function Sidebar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    await logout();
    navigate("/login");
  };

  const firstName = user?.fullName?.split(" ")?.[0] || "Friend";
  const initials = (user?.fullName || "U")[0]?.toUpperCase();

  return (
    <Box
      sx={{
        borderRadius: 4,
        overflow: "hidden",
        p: 2.3,

        height: { xs: "auto", md: "calc(100vh - 48px)" },
        position: { xs: "relative", md: "sticky" },
        top: { md: 18 },

        background:
          "linear-gradient(180deg, rgba(255,255,255,0.78), rgba(243,241,248,0.48))",
        border: "1px solid rgba(15,23,42,0.04)",
        backdropFilter: "blur(16px)",
        boxShadow: "0 16px 40px rgba(15,23,42,0.08)",

        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* ✅ Header */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <Typography
          sx={{
            fontSize: 22,
            fontWeight: 1000,
            color: "#2F8790",
            letterSpacing: 0.2,
          }}
        >
          Krishna’s Lens
        </Typography>

        <Box
          sx={{
            px: 1.2,
            py: 0.45,
            borderRadius: 999,
            fontSize: 12,
            fontWeight: 900,
            color: "#6B7A85",
            background: "rgba(255,255,255,0.62)",
            border: "1px solid rgba(15,23,42,0.06)",
          }}
        >
          Recurras
        </Box>
      </Box>

      {/* ✅ Profile */}
      <Box sx={{ mt: 2.1, display: "flex", gap: 1.2, alignItems: "center" }}>
        <Box
          sx={{
            width: 50,
            height: 50,
            borderRadius: "50%",
            background: "rgba(74,163,184,0.14)",
            border: "1px solid rgba(15,23,42,0.04)",
            display: "grid",
            placeItems: "center",
            fontWeight: 1000,
            color: "#2F8790",
            userSelect: "none",
            flexShrink: 0,
          }}
        >
          {initials}
        </Box>

        <Box sx={{ minWidth: 0 }}>
          <Typography sx={{ fontWeight: 1000, lineHeight: 1.1 }} noWrap>
            {user?.fullName || "User"}
          </Typography>
          <Typography variant="caption" sx={{ color: "text.secondary" }}>
            Welcome back, {firstName}!
          </Typography>
        </Box>
      </Box>

      {/* ✅ Nav */}
      <Box
        sx={{
          mt: 2.7,
          borderRadius: 3,
          p: 1,
          background: "rgba(255,255,255,0.42)",
          border: "1px solid rgba(15,23,42,0.04)",
        }}
      >
        {navItems.map((item) => (
          <NavLink key={item.to} to={item.to} style={{ textDecoration: "none" }}>
            {({ isActive }) => (
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: 1.1,
                  px: 1.25,
                  py: 1.05,
                  borderRadius: 3,
                  transition: "0.18s",
                  color: isActive ? "#0F172A" : "#52606D",
                  background: isActive
                    ? "linear-gradient(90deg, rgba(74,163,184,0.20), rgba(47,135,144,0.08))"
                    : "transparent",
                  border: isActive
                    ? "1px solid rgba(74,163,184,0.18)"
                    : "1px solid transparent",
                  "&:hover": { background: "rgba(255,255,255,0.62)" },
                }}
              >
                <Box sx={{ opacity: isActive ? 1 : 0.7 }}>{item.icon}</Box>

                <Typography sx={{ fontWeight: 900, fontSize: 14 }}>
                  {item.label}
                </Typography>

                <Box sx={{ flex: 1 }} />
                <Typography sx={{ opacity: 0.22, fontWeight: 1000 }}>›</Typography>
              </Box>
            )}
          </NavLink>
        ))}
      </Box>

      {/* ✅ Logout */}
      <Box sx={{ mt: 2.6 }}>
        <Button
          fullWidth
          onClick={handleLogout}
          startIcon={<LogoutRoundedIcon />}
          sx={{
            borderRadius: 3,
            py: 1.05,
            fontWeight: 1000,
            textTransform: "none",
            color: "white",
            background: "linear-gradient(90deg, #4AA3B8, #2F8790)",
            boxShadow: "0 12px 24px rgba(74,163,184,0.22)",
            "&:hover": {
              background: "linear-gradient(90deg, #2F8790, #4AA3B8)",
            },
          }}
        >
          Logout
        </Button>
      </Box>

      {/* ✅ FINAL BOTTOM SECTION (Same position, better glow) */}
      <Box
        sx={{
          mt: 2.6,
          flexGrow: 1,
          borderRadius: 4,
          position: "relative",
          overflow: "hidden",
          display: { xs: "none", md: "block" },

          border: "1px solid rgba(15,23,42,0.06)",
          background:
            "linear-gradient(180deg, rgba(255,255,255,0.72), rgba(243,241,248,0.30))",
          boxShadow: "0 18px 50px rgba(15,23,42,0.08)",
        }}
      >
        {/* ✅ base aura */}
        <Box
          sx={{
            position: "absolute",
            inset: -90,
            background:
              "radial-gradient(circle at 22% 18%, rgba(74,163,184,0.32), transparent 62%), radial-gradient(circle at 80% 86%, rgba(59,130,246,0.12), transparent 60%)",
            filter: "blur(26px)",
            opacity: 1,
            pointerEvents: "none",
          }}
        />

        {/* ✅ EXTRA feather glow boost (makes it dashboard-like) */}
        <Box
          sx={{
            position: "absolute",
            inset: 0,
            background:
              "radial-gradient(circle at 70% 52%, rgba(74,163,184,0.30), transparent 58%)",
            filter: "blur(70px)",
            opacity: 0.95,
            pointerEvents: "none",
          }}
        />

        {/* ✅ Feather cover (FULL, strong glow) */}
        <Box
          component="img"
          src={featherGlow}
          alt="Feather glow"
          sx={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            objectFit: "contain",
            objectPosition: "center",
            opacity: 0.88, // ✅ more visible
            transform: "rotate(-8deg) scale(1.06)",

            pointerEvents: "none",
            filter:
              "drop-shadow(0 30px 80px rgba(47,135,144,0.35)) drop-shadow(0 60px 120px rgba(74,163,184,0.22)) saturate(1.12) contrast(1.06)",
          }}
        />

        {/* ✅ Krishna glow watermark */}
        <Box
          component="img"
          src={krishnaGlow}
          alt="Krishna glow"
          sx={{
            position: "absolute",
            left: "50%",
            bottom: -22,
            transform: "translateX(-50%)",
            width: 320,
            maxWidth: "95%",
            opacity: 0.18, // ✅ thoda kam so feather dominates
            pointerEvents: "none",
            filter: "drop-shadow(0 55px 110px rgba(0,0,0,0.14))",
          }}
        />

        {/* ✅ top glass fade (reduced so glow not killed) */}
        <Box
          sx={{
            position: "absolute",
            inset: 0,
            background:
              "linear-gradient(180deg, rgba(255,255,255,0.62), rgba(255,255,255,0.10))",
            opacity: 0.35, // ✅ reduced
            pointerEvents: "none",
          }}
        />
      </Box>
    </Box>
  );
}
