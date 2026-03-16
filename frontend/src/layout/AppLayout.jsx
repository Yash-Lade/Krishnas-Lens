import { Box } from "@mui/material";
import Sidebar from "./Sidebar";
import Topbar from "./Topbar";
import bg from "../assets/images/bg-watercolor-clouds.png";
import mandala from "../assets/images/mandala-tile.png";
import { Outlet } from "react-router-dom";

export default function AppLayout() {
  return (
    <Box
      sx={{
        minHeight: "100vh",
        width: "100%",
        position: "relative",
        backgroundImage: `url(${bg})`,
        backgroundSize: "cover",
        backgroundPosition: "center",
        overflow: "hidden",
      }}
    >
      {/* Mandala overlay */}
      <Box
        sx={{
          position: "absolute",
          inset: 0,
          backgroundImage: `url(${mandala})`,
          backgroundRepeat: "repeat",
          backgroundSize: "520px",
          opacity: 0.02,
          pointerEvents: "none",
          zIndex: 0,
        }}
      />

      {/* Soft overlay */}
      <Box
        sx={{
          position: "absolute",
          inset: 0,
          background:
            "linear-gradient(180deg, rgba(255,255,255,0.70), rgba(255,255,255,0.45))",
          pointerEvents: "none",
          zIndex: 0,
        }}
      />

      {/* Main layout */}
      <Box
        sx={{
          position: "relative",
          zIndex: 2,
          maxWidth: 1240,
          mx: "auto",
          px: { xs: 1.5, md: 3 },
          py: { xs: 1.5, md: 3 },
          display: "grid",
          gridTemplateColumns: { xs: "1fr", md: "300px 1fr" },
          gap: { xs: 1.5, md: 3 },
          alignItems: "start",
        }}
      >
        <Sidebar />

        <Box sx={{ display: "grid", gap: 2, minWidth: 0 }}>
          <Topbar />

          {/* ✅ Outlet render */}
          <Box sx={{ width: "100%", minWidth: 0 }}>
            <Outlet />
          </Box>
        </Box>
      </Box>
    </Box>
  );
}
