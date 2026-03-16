import { Routes, Route } from "react-router-dom";
import LandingPage from "../pages/landing/LandingPage";
import LoginPage from "../pages/auth/LoginPage";
import SignupPage from "../pages/auth/SignupPage";

import DashboardPage from "../pages/dashboard/DashboardPage";
import SubmitThoughtPage from "../pages/entries/SubmitThoughtPage";
import EntryResultPage from "../pages/entries/EntryResultPage";
import HistoryPage from "../pages/entries/HistoryPage";
import SettingsPage from "../pages/settings/SettingsPage";

import AppLayout from "../layout/AppLayout";
import PrivateRoute from "./PrivateRoute";

export default function AppRoutes() {
  return (
    <Routes>
      {/* Public */}
      <Route path="/" element={<LandingPage />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/signup" element={<SignupPage />} />

      {/* Protected Layout */}
      <Route
        element={
          <PrivateRoute>
            <AppLayout />
          </PrivateRoute>
        }
      >
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/submit" element={<SubmitThoughtPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/entry/:entryId" element={<EntryResultPage />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Route>
    </Routes>
  );
}
