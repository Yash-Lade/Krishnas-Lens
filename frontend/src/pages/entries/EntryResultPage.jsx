import { useEffect, useState } from "react";
import { Box, Typography, Button, Chip } from "@mui/material";
import { useNavigate, useParams } from "react-router-dom";
import Loader from "../../components/common/Loader";
import LensOutput from "../../components/entries/LensOutput";
import { getEntryByIdApi } from "../../services/entryApi";

export default function EntryResultPage() {
  const { entryId } = useParams();
  const navigate = useNavigate();

  const [loading, setLoading] = useState(true);
  const [entry, setEntry] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        const res = await getEntryByIdApi(entryId);
        setEntry(res?.data?.data || res?.data);
      } finally {
        setLoading(false);
      }
    })();
  }, [entryId]);

  if (loading) return <Loader text="Loading lens output..." />;

  return (
    <Box sx={{ display: "grid", gap: 2 }}>
      {/* summary card */}
      <Box
        sx={{
          borderRadius: 4,
          p: { xs: 2, md: 3 },
          background:
            "linear-gradient(180deg, rgba(255,255,255,0.80), rgba(243,241,248,0.48))",
          border: "1px solid rgba(15,23,42,0.04)",
          backdropFilter: "blur(16px)",
          boxShadow: "0 16px 40px rgba(15,23,42,0.08)",
        }}
      >
        <Typography sx={{ fontSize: { xs: 20, md: 26 }, fontWeight: 1000 }}>
          Krishna’s Lens Output
        </Typography>

        <Typography sx={{ mt: 1, color: "text.secondary", lineHeight: 1.7 }}>
          {entry?.thoughtText}
        </Typography>

        <Box sx={{ mt: 2, display: "flex", gap: 1, flexWrap: "wrap" }}>
          <Chip
            label={`Mood: ${(entry?.mood || "calm").toUpperCase()}`}
            sx={{ borderRadius: 999, fontWeight: 1000, background: "rgba(74,163,184,0.14)" }}
          />
          <Chip
            label={`Severity: ${(entry?.severity || "low").toUpperCase()}`}
            sx={{ borderRadius: 999, fontWeight: 1000, background: "rgba(15,23,42,0.06)" }}
          />
        </Box>

        {/* actions */}
        <Box sx={{ mt: 2, display: "flex", gap: 1, flexWrap: "wrap" }}>
          <Button
            onClick={() => navigate("/submit")}
            variant="contained"
            sx={{
              borderRadius: 4,
              fontWeight: 1000,
              background: "linear-gradient(90deg, #4AA3B8, #2F8790)",
              boxShadow: "0 14px 28px rgba(74,163,184,0.18)",
            }}
          >
            Submit Another
          </Button>

          <Button
            onClick={() => navigate("/history")}
            variant="outlined"
            sx={{ borderRadius: 4, fontWeight: 900 }}
          >
            View History
          </Button>

          <Button
            onClick={() => navigate("/dashboard")}
            sx={{ borderRadius: 4, fontWeight: 900 }}
          >
            Back to Dashboard
          </Button>
        </Box>
      </Box>

      {/* lens outputs */}
      <LensOutput data={entry} />
    </Box>
  );
}
