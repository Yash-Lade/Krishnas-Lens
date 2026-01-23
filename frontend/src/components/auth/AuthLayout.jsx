import React from "react";
import { Box, Container, Paper } from "@mui/material";

import bgClouds from "../../assets/images/bg-watercolor-clouds.png";
import mandala from "../../assets/images/mandala-tile.png";

export default function AuthLayout({ children }) {
  return (
    <Box
      sx={{
        minHeight: "100vh",
        position: "relative",
        overflowX: "hidden",
        background:
          "linear-gradient(180deg, #FBFDFF 0%, #EEF4FF 45%, #F3F1F8 100%)",
        display: "flex",
        alignItems: "center",
        py: { xs: 3, md: 5 },
      }}
    >
      {/* Background watercolor */}
      <Box
        sx={{
          position: "absolute",
          inset: 0,
          backgroundImage: `url(${bgClouds})`,
          backgroundSize: "cover",
          backgroundPosition: "center",
          opacity: 0.55,
          pointerEvents: "none",
        }}
      />

      {/* Mandala overlay */}
      <Box
        sx={{
          position: "absolute",
          top: { xs: "-180px", md: "-120px" },
          right: { xs: "-220px", md: "-180px" },
          width: { xs: 460, md: 720 },
          height: { xs: 460, md: 720 },
          backgroundImage: `url(${mandala})`,
          backgroundSize: "cover",
          opacity: 0.05,
          pointerEvents: "none",
        }}
      />

      <Container maxWidth="lg" sx={{ position: "relative", zIndex: 2 }}>
        <Paper
          elevation={0}
          sx={{
            borderRadius: { xs: 5, md: 6 },
            overflow: "hidden",
            position: "relative", // ✅ IMPORTANT
            border: "1px solid rgba(255,255,255,0.60)",
            background:
              "linear-gradient(180deg, rgba(255,255,255,0.66), rgba(243,241,248,0.46))",
            backdropFilter: "blur(16px)",
            boxShadow: "0 18px 45px rgba(15,23,42,0.12)",
          }}
        >
          {/* ✅ Premium glow layers (Dashboard style) */}
          <Box
            sx={{
              position: "absolute",
              right: -180,
              top: -180,
              width: 560,
              height: 560,
              borderRadius: "50%",
              background: "rgba(74,163,184,0.26)",
              filter: "blur(120px)",
              opacity: 0.95,
              pointerEvents: "none",
            }}
          />
          <Box
            sx={{
              position: "absolute",
              left: -230,
              bottom: -260,
              width: 620,
              height: 620,
              borderRadius: "50%",
              background: "rgba(15,23,42,0.08)",
              filter: "blur(140px)",
              opacity: 0.55,
              pointerEvents: "none",
            }}
          />
          <Box
            sx={{
              position: "absolute",
              inset: 0,
              background:
                "radial-gradient(circle at 70% 40%, rgba(74,163,184,0.18), transparent 58%)",
              filter: "blur(50px)",
              opacity: 0.9,
              pointerEvents: "none",
            }}
          />

          {/* ✅ Ensure content above glow */}
          <Box sx={{ position: "relative", zIndex: 2 }}>{children}</Box>
        </Paper>
      </Container>
    </Box>
  );
}
