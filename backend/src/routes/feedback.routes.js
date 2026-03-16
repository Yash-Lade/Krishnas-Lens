import { Router } from "express";
import { verifyJWT } from "../middlewares/auth.middlewares.js";

import {
  createFeedback,
  getMyFeedback,
  deleteFeedback,
} from "../controllers/feedback.controllers.js";

const router = Router();

// ✅ All feedback routes are protected
router.use(verifyJWT);

router.post("/", createFeedback);
router.get("/my", getMyFeedback);
router.delete("/:feedbackId", deleteFeedback);

export default router;
