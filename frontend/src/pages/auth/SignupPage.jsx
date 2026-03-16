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
import { registerUserApi } from "../../services/authApi";
import useSnackbar from "../../hooks/useSnackbar";

export default function SignupPage() {
  const navigate = useNavigate();

  const [showPass, setShowPass] = useState(false);
  const [loading, setLoading] = useState(false);

  const { showSnackbar } = useSnackbar();

  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const [form, setForm] = useState({
    fullName: "",
    email: "",
    password: "",
  });

  const onChange = (e) =>
    setForm((p) => ({
      ...p,
      [e.target.name]: e.target.value,
    }));

  // ✅ simple client validation
  const validate = () => {
    if (!form.fullName.trim()) return "Full name is required";
    if (!form.email.trim()) return "Email is required";
    if (!/^\S+@\S+\.\S+$/.test(form.email)) return "Enter a valid email";
    if (!form.password.trim()) return "Password is required";
    if (form.password.length < 6)
      return "Password must be at least 6 characters";
    return "";
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    const msg = validate();
    if (msg) {
      setError(msg);
      return;
    }

    setLoading(true);
    try {
      await registerUserApi({
        fullName: form.fullName,
        email: form.email,
        password: form.password,
      });

      showSnackbar("Account created ✅ Please login", "success");
      setTimeout(() => navigate("/login"), 800);
    } catch (err) {
      const msg =
        err?.response?.data?.message || err?.message || "Signup failed";
      showSnackbar(msg, "error");
    } finally {
      setLoading(false);
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
          title="Create Account ✨"
          subtitle="Start your journey towards clarity, calmness, and consistency."
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
            Sign Up
          </Typography>

          <Typography sx={{ color: "text.secondary", fontSize: 14, mb: 2.5 }}>
            Create your account to save thoughts, moods & reflections.
          </Typography>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {success && (
            <Alert severity="success" sx={{ mb: 2 }}>
              {success}
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
              label="Full Name"
              name="fullName"
              value={form.fullName}
              onChange={onChange}
              fullWidth
            />

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
              {loading ? "Creating..." : "Create Account"}
            </Button>

            <Divider sx={{ opacity: 0.4 }} />

            <Stack
              direction="row"
              justifyContent="center"
              spacing={1}
              sx={{ fontSize: 14 }}
            >
              <Typography sx={{ color: "text.secondary" }}>
                Already have an account?
              </Typography>
              <Typography
                component={Link}
                to="/login"
                sx={{ color: "#2F8790", fontWeight: 800 }}
              >
                Login
              </Typography>
            </Stack>
          </Box>
        </Box>
      </Box>
    </AuthLayout>
  );
}
