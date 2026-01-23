import { createContext, useContext, useEffect, useState } from "react";
import { loginUserApi, logoutUserApi } from "../services/authApi";

export const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(() => {
    const stored = localStorage.getItem("kl_user");
    return stored ? JSON.parse(stored) : null;
  });

  const [loading, setLoading] = useState(false);

  const login = async (email, password) => {
    setLoading(true);
    try {
      const res = await loginUserApi({ email, password });
      const data = res?.data?.data || res?.data;

      if (data?.accessToken) {
        localStorage.setItem("kl_access_token", data.accessToken);
      }

      if (data?.user) {
        localStorage.setItem("kl_user", JSON.stringify(data.user));
        setUser(data.user);
      }

      return data;
    } finally {
      setLoading(false);
    }
  };

  const logout = async () => {
    try {
      await logoutUserApi();
    } catch (e) {
      // ignore
    } finally {
      localStorage.removeItem("kl_user");
      localStorage.removeItem("kl_access_token");
      setUser(null);
    }
  };

  return (
    <AuthContext.Provider value={{ user, setUser, loading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuthContext = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuthContext must be used inside AuthProvider");
  return ctx;
};
