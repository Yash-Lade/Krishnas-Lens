import { Box, Typography } from "@mui/material";

export default function LensCard({ title, icon, children, accent = "#4AA3B8" }) {
  return (
    <Box
      sx={{
        borderRadius: 4,
        p: 2.3,
        background:
          "linear-gradient(180deg, rgba(255,255,255,0.70), rgba(243,241,248,0.46))",
        border: "1px solid rgba(255,255,255,0.62)",
        backdropFilter: "blur(18px)",
        boxShadow: "0 18px 45px rgba(15,23,42,0.10)",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
        <Box
          sx={{
            width: 38,
            height: 38,
            borderRadius: 2,
            display: "grid",
            placeItems: "center",
            background: "rgba(74,163,184,0.12)",
            color: accent,
          }}
        >
          {icon}
        </Box>
        <Typography variant="h6" sx={{ fontWeight: 700 }}>
          {title}
        </Typography>
      </Box>

      <Box sx={{ mt: 1.4 }}>{children}</Box>
    </Box>
  );
}
