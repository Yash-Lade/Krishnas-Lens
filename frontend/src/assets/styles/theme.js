import { createTheme } from "@mui/material/styles";

const theme = createTheme({
  palette: {
    primary: { main: "#4AA3B8" },
    secondary: { main: "#2F8790" },
    text: { primary: "#213139", secondary: "#6B7A85" },
    background: { default: "#EFF3FA" },
  },
  typography: {
    fontFamily: '"Inter", "Poppins", system-ui, -apple-system, Segoe UI, Roboto, Arial',
    button: { textTransform: "none", fontWeight: 800 },
  },

  // consistent radii system
  shape: { borderRadius: 12 },

  // single shadow system
  shadows: [
    "none",
    "0 6px 18px rgba(15,23,42,0.06)",
    "0 10px 28px rgba(15,23,42,0.08)",
    /* ... default MUI entries if you want ... */
  ],

  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          backgroundClip: "padding-box",
        },
      },
    },

    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: "0 10px 28px rgba(15,23,42,0.06)",
          border: "1px solid rgba(15,23,42,0.04)",
        },
      },
    },

    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          background: "rgba(255,255,255,0.6)",
          backdropFilter: "blur(10px)",
          border: "1px solid rgba(15,23,42,0.04)",
        },
        input: { padding: "14px 14px" },
        notchedOutline: { border: "none" },
      },
    },

    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 14,
          boxShadow: "0 10px 26px rgba(74,163,184,0.15)",
          fontWeight: 800,
        },
      },
    },

    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 999,
          fontWeight: 800,
        },
      },
    },
  },
});

export default theme;
