import React from "react";
import { Box, Typography, Chip } from "@mui/material";
import heroKrishna from "../../assets/images/hero-krishna-feather.png";

export default function AuthQuotePanel({ title, subtitle }) {
  return (
    <Box
      sx={{
        position: "relative",
        height: "100%",
        p: { xs: 3, md: 5 },
        display: { xs: "none", md: "flex" },
        flexDirection: "column",
        justifyContent: "space-between",
        borderRight: "1px solid rgba(255,255,255,0.55)",
      }}
    >
      <Box>
        <Chip
          label="Krishna’s Lens"
          sx={{
            bgcolor: "rgba(74,163,184,0.14)",
            color: "#2F8790",
            fontWeight: 800,
            mb: 2,
          }}
        />

        <Typography
          sx={{
            fontFamily: "Poppins",
            fontWeight: 800,
            fontSize: 26,
            color: "#213139",
            mb: 1,
          }}
        >
          {title}
        </Typography>

        <Typography sx={{ color: "text.secondary", fontSize: 14, maxWidth: 360 }}>
          {subtitle}
        </Typography>
      </Box>

      {/* Quote */}
      <Box
        sx={{
          mt: 3,
          p: 2.2,
          borderRadius: 4,
          border: "1px solid rgba(255,255,255,0.55)",
          background: "rgba(255,255,255,0.52)",
          backdropFilter: "blur(12px)",
        }}
      >
        <Typography sx={{ fontStyle: "italic", color: "#213139", fontSize: 14, mb: 1 }}>
          “A calm mind sees the right path.”
        </Typography>
        <Typography sx={{ color: "text.secondary", fontSize: 12 }}>
          — Inspired by Bhagavad Gita
        </Typography>
      </Box>

      {/* Illustration */}
      <Box
        component="img"
        src={heroKrishna}
        alt="Krishna illustration"
        sx={{
          width: "92%",
          maxWidth: 420,
          alignSelf: "center",
          opacity: 0.95,
          filter: "drop-shadow(0 18px 35px rgba(15,23,42,0.12))",
          mt: 2,
        }}
      />
    </Box>
  );
}
