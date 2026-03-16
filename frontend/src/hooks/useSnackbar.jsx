import { createContext, useContext, useMemo, useState } from "react";
import { Alert, Snackbar } from "@mui/material";

const SnackbarContext = createContext(null);

export function SnackbarProvider({ children }) {
  const [snack, setSnack] = useState({
    open: false,
    message: "",
    severity: "success",
    duration: 3000,
  });

  const showSnackbar = (message, severity = "success", duration = 3000) => {
    setSnack({ open: true, message, severity, duration });
  };

  const handleClose = (_, reason) => {
    if (reason === "clickaway") return;
    setSnack((prev) => ({ ...prev, open: false }));
  };

  const value = useMemo(() => ({ showSnackbar }), []);

  return (
    <SnackbarContext.Provider value={value}>
      {children}

      <Snackbar
        open={snack.open}
        autoHideDuration={snack.duration}
        onClose={handleClose}
        anchorOrigin={{ vertical: "top", horizontal: "right" }}
      >
        <Alert
          onClose={handleClose}
          severity={snack.severity}
          variant="filled"
          sx={{
            borderRadius: 3,
            fontWeight: 700,
            boxShadow: "0 18px 40px rgba(0,0,0,0.18)",
          }}
        >
          {snack.message}
        </Alert>
      </Snackbar>
    </SnackbarContext.Provider>
  );
}

export default function useSnackbar() {
  const ctx = useContext(SnackbarContext);
  if (!ctx) throw new Error("useSnackbar must be used within SnackbarProvider");
  return ctx;
}
