import { Router } from "express";
import {
  registerUser,
  loginUser,
  refreshAcessToken,
  logoutUser,
  updateProfile,
} from "../controllers/user.controllers.js";

import { verifyJWT } from "../middlewares/auth.middlewares.js";

const router = Router();

// ✅ Auth routes
router.post("/register", registerUser);
router.post("/login", loginUser);
router.post("/refresh-token", refreshAcessToken);

// ✅ Protected routes
router.post("/logout", verifyJWT, logoutUser);
router.patch("/profile", verifyJWT, updateProfile);

export default router;
