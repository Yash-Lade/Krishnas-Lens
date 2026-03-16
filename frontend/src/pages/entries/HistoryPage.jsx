import { useEffect, useMemo, useState } from "react";
import {
  Box,
  TextField,
  Typography,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Chip,
} from "@mui/material";
import Loader from "../../components/common/Loader";
import EntryTable from "../../components/entries/EntryTable";
import useSnackbar from "../../hooks/useSnackbar";
import { deleteEntryApi, getMyEntriesApi } from "../../services/entryApi";
import { useNavigate } from "react-router-dom";

export default function HistoryPage() {
  const { showSnackbar } = useSnackbar();
  const navigate = useNavigate();

  const [loading, setLoading] = useState(true);
  const [entries, setEntries] = useState([]);
  const [q, setQ] = useState("");

  // delete dialog
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState(null);

  const filtered = useMemo(() => {
    const query = q.trim().toLowerCase();
    if (!query) return entries;
    return entries.filter((e) =>
      (e.thoughtText || "").toLowerCase().includes(query)
    );
  }, [q, entries]);

  const load = async () => {
    try {
      setLoading(true);
      const res = await getMyEntriesApi();
      setEntries(res?.data?.data || res?.data || []);
    } catch {
      showSnackbar("Failed to fetch entries", "error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const handleView = (e) => navigate(`/entry/${e._id}`);

  const askDelete = (e) => {
    setSelected(e);
    setOpen(true);
  };

  const confirmDelete = async () => {
    try {
      await deleteEntryApi(selected._id);
      showSnackbar("Entry deleted ✅", "success");
      setOpen(false);
      setSelected(null);
      load();
    } catch {
      showSnackbar("Delete failed", "error");
    }
  };

  if (loading) return <Loader text="Fetching history..." />;

  return (
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
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 2, mb: 2 }}>
        <Box>
          <Typography sx={{ fontSize: { xs: 20, md: 26 }, fontWeight: 1000 }}>
            History
          </Typography>
          <Typography sx={{ color: "text.secondary" }}>
            View and manage your thoughts.
          </Typography>
        </Box>

        <Chip
          label={`${entries.length} entries`}
          sx={{
            fontWeight: 1000,
            borderRadius: 999,
            background: "rgba(15,23,42,0.05)",
          }}
        />
      </Box>

      <TextField
        value={q}
        onChange={(e) => setQ(e.target.value)}
        fullWidth
        placeholder="Search entries..."
        sx={{
          mb: 2,
          "& .MuiOutlinedInput-root": { borderRadius: 4 },
        }}
      />

      <EntryTable rows={filtered} onView={handleView} onDelete={askDelete} />

      {/* Delete confirmation */}
      <Dialog open={open} onClose={() => setOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ fontWeight: 1000 }}>Delete Entry?</DialogTitle>
        <DialogContent sx={{ color: "text.secondary" }}>
          Are you sure you want to delete this entry? This action cannot be undone.
          <Box sx={{ mt: 1, fontWeight: 900, color: "text.primary" }}>
            “{selected?.thoughtText?.slice(0, 80)}{selected?.thoughtText?.length > 80 ? "..." : ""}”
          </Box>
        </DialogContent>
        <DialogActions sx={{ p: 2 }}>
          <Button onClick={() => setOpen(false)} sx={{ fontWeight: 900 }}>
            Cancel
          </Button>
          <Button onClick={confirmDelete} color="error" variant="contained" sx={{ fontWeight: 1000, borderRadius: 3 }}>
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
