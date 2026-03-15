import { useState } from "react";
import { Box, Button, TextField, Typography } from "@mui/material";
import peacock from "../../assets/images/peacock-feather.png";
import useSnackbar from "../../hooks/useSnackbar";
import { updateProfileApi } from "../../services/authApi";
import { useAuth } from "../../hooks/useAuth";
import FeedbackForm from "../../components/feedback/FeedbackForm";


export default function SettingsPage() {
  const { showSnackbar } = useSnackbar();
  const { user, setUser } = useAuth();

  const [loading, setLoading] = useState(false);
  const [fullName, setFullName] = useState(user?.fullName || "");
  const [email] = useState(user?.email || "");

  const onSave = async () => {
    try {
      setLoading(true);
      const res = await updateProfileApi({ fullName });
      const updated = res?.data?.data || res?.data;

      // update local auth user
      const newUser = { ...user, fullName: updated?.fullName || fullName };
      localStorage.setItem("kl_user", JSON.stringify(newUser));
      setUser(newUser);

      showSnackbar("Profile updated ✅", "success");
    } catch (err) {
      showSnackbar("Update failed", "error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ position: "relative" }}>
      <Box
        component="img"
        src={peacock}
        alt="peacock"
        sx={{
          position: "absolute",
          right: -18,
          top: -8,
          width: { xs: 150, md: 220 },
          opacity: 0.85,
          pointerEvents: "none",
        }}
      />

      <Box
        sx={{
          borderRadius: 4,
          p: { xs: 2, md: 3 },
          background:
            "linear-gradient(180deg, rgba(255,255,255,0.70), rgba(243,241,248,0.46))",
          border: "1px solid rgba(255,255,255,0.62)",
          backdropFilter: "blur(18px)",
          boxShadow: "0 18px 45px rgba(15,23,42,0.10)",
        }}
      >
        <Typography variant="h5" sx={{ fontWeight: 900, mb: 0.5 }}>
          Account Settings
        </Typography>
        <Typography sx={{ color: "text.secondary", mb: 2 }}>
          Manage your profile and preferences.
        </Typography>

        <Box sx={{ display: "grid", gap: 2 }}>
          <Box>
            <Typography sx={{ fontWeight: 700, mb: 0.7 }}>Full Name</Typography>
            <TextField
              fullWidth
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              sx={{
                "& .MuiOutlinedInput-root": {
                  borderRadius: 3,
                  background: "rgba(255,255,255,0.55)",
                },
              }}
            />
          </Box>

          <Box>
            <Typography sx={{ fontWeight: 700, mb: 0.7 }}>Email</Typography>
            <TextField
              fullWidth
              value={email}
              disabled
              sx={{
                "& .MuiOutlinedInput-root": {
                  borderRadius: 3,
                  background: "rgba(255,255,255,0.40)",
                },
              }}
            />
          </Box>

          <Button
            disabled={loading}
            onClick={onSave}
            variant="contained"
            sx={{
              borderRadius: 3,
              textTransform: "none",
              fontWeight: 800,
              py: 1.1,
              background:
                "linear-gradient(90deg, rgba(74,163,184,1), rgba(47,135,144,1))",
              boxShadow: "0 12px 26px rgba(74,163,184,0.32)",
            }}
          >
            Save Changes
          </Button>
        </Box>
      </Box>
    </Box>
    
  );
}
