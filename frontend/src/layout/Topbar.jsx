import { Box, Button, Typography, TextField, InputAdornment } from "@mui/material";
import AddRoundedIcon from "@mui/icons-material/AddRounded";
import SearchRoundedIcon from "@mui/icons-material/SearchRounded";
import { useLocation, useNavigate } from "react-router-dom";

export default function Topbar() {
  const navigate = useNavigate();
  const { pathname } = useLocation();

  const title =
    pathname === "/dashboard"
      ? "Dashboard"
      : pathname === "/submit"
      ? "Submit a Thought"
      : pathname === "/history"
      ? "History"
      : pathname === "/settings"
      ? "Settings"
      : "Krishna’s Lens";

  return (
    <Box
      sx={{
        borderRadius: 4,
        px: { xs: 2, md: 2.6 },
        py: 1.3,
        background: "rgba(255,255,255,0.72)",
        border: "1px solid rgba(15,23,42,0.06)",
        backdropFilter: "blur(14px)",
        boxShadow: "0 16px 35px rgba(15,23,42,0.08)",

        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 2,

        // ✅ Fix layout squeeze in smaller widths
        flexWrap: { xs: "wrap", md: "nowrap" },
      }}
    >
      <Box sx={{ minWidth: 160 }}>
        <Typography sx={{ fontSize: { xs: 20, md: 26 }, fontWeight: 1000 }}>
          {title}
        </Typography>
      </Box>

      {/* ✅ Search bar (hide on mobile) */}
      <Box sx={{ flex: 1, minWidth: 260, display: { xs: "none", md: "block" } }}>
        <TextField
          fullWidth
          placeholder="Search entries..."
          size="small"
          sx={{
            "& .MuiOutlinedInput-root": {
              borderRadius: 999,
              background: "rgba(255,255,255,0.65)",
            },
          }}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchRoundedIcon sx={{ opacity: 0.55 }} />
              </InputAdornment>
            ),
          }}
        />
      </Box>

      <Button
        onClick={() => navigate("/submit")}
        variant="contained"
        startIcon={<AddRoundedIcon />}
        sx={{
          borderRadius: 999,
          textTransform: "none",
          fontWeight: 1000,
          px: 2.2,
          py: 1.05,
          background: "linear-gradient(90deg, #4AA3B8, #2F8790)",
          boxShadow: "0 14px 28px rgba(74,163,184,0.22)",
        }}
      >
        Submit a Thought
      </Button>
    </Box>
  );
}
