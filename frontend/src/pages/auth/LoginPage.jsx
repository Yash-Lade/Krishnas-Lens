import React, { useState } from "react";
import {
  Box,
  Typography,
  TextField,
  Button,
  Stack,
  IconButton,
  InputAdornment,
  Divider,
  Alert,
} from "@mui/material";
import VisibilityRoundedIcon from "@mui/icons-material/VisibilityRounded";
import VisibilityOffRoundedIcon from "@mui/icons-material/VisibilityOffRounded";
import { Link, useNavigate } from "react-router-dom";
import AuthLayout from "../../components/auth/AuthLayout";
import AuthQuotePanel from "../../components/auth/AuthQuotePanel";
import { useAuthContext } from "../../context/AuthContext";
import useSnackbar from "../../hooks/useSnackbar";


export default function LoginPage() {
  const navigate = useNavigate();
  const { login, loading } = useAuthContext();

  const { showSnackbar } = useSnackbar();

  const [showPass, setShowPass] = useState(false);
  const [error, setError] = useState("");

  const [form, setForm] = useState({ email: "", password: "" });
  const onChange = (e) =>
    setForm((p) => ({ ...p, [e.target.name]: e.target.value }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!form.email || !form.password) {
      setError("Email and password are required");
      return;
    }
    try {
      await login(form.email, form.password);
      showSnackbar("Login successful ✅", "success");
      navigate("/dashboard");
    } catch (err) {
      showSnackbar(err.message || "Login failed", "error");
    }
  };

  return (
    <AuthLayout>
      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: { xs: "1fr", md: "1fr 1fr" },
        }}
      >
        <AuthQuotePanel
          title="Welcome Back 👋"
          subtitle="Sign in to continue tracking thoughts, moods, and insights."
        />

        {/* Right form */}
  <Box sx={{ p: { xs: 3, sm: 4, md: 5 }, position: "relative", zIndex: 2 }}>
          <Typography
            sx={{
              fontFamily: "Poppins",
              fontWeight: 800,
              fontSize: { xs: 22, md: 26 },
              color: "#213139",
              mb: 0.5,
            }}
          >
            Login
          </Typography>

          <Typography sx={{ color: "text.secondary", fontSize: 14, mb: 2.5 }}>
            Enter your credentials to access your dashboard.
          </Typography>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          <Box
            component="form"
            onSubmit={handleSubmit}
            sx={{
              display: "grid",
              gap: 2,
              maxWidth: 520,
            }}
          >
            <TextField
              label="Email"
              name="email"
              value={form.email}
              onChange={onChange}
              fullWidth
            />

            <TextField
              label="Password"
              name="password"
              value={form.password}
              onChange={onChange}
              type={showPass ? "text" : "password"}
              fullWidth
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      onClick={() => setShowPass((s) => !s)}
                      edge="end"
                    >
                      {showPass ? (
                        <VisibilityOffRoundedIcon />
                      ) : (
                        <VisibilityRoundedIcon />
                      )}
                    </IconButton>
                  </InputAdornment>
                ),
              }}
            />

            <Button
              type="submit"
              variant="contained"
              disabled={loading}
              sx={{
                bgcolor: "#4AA3B8",
                py: 1.2,
                boxShadow: "0 14px 28px rgba(74,163,184,0.24)",
                "&:hover": { bgcolor: "#2F8790" },
              }}
            >
              {loading ? "Logging in..." : "Login"}
            </Button>

            <Divider sx={{ opacity: 0.4 }} />

            <Stack
              direction="row"
              justifyContent="center"
              spacing={1}
              sx={{ fontSize: 14 }}
            >
              <Typography sx={{ color: "text.secondary" }}>
                Don’t have an account?
              </Typography>
              <Typography
                component={Link}
                to="/signup"
                sx={{ color: "#2F8790", fontWeight: 800 }}
              >
                Sign Up
              </Typography>
            </Stack>
          </Box>
        </Box>
      </Box>
    </AuthLayout>
  );
}
