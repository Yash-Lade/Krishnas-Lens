import { Router } from "express";
import { verifyJWT } from "../middlewares/auth.middlewares.js";

import {
  createEntry,
  getMyEntries,
  getEntryById,
  updateEntry,
  deleteEntry,
} from "../controllers/entry.controllers.js";

const router = Router();

// ✅ All entry routes are protected
router.use(verifyJWT);

// Create + List
router.route("/").post(createEntry).get(getMyEntries);

// Single entry
router
  .route("/:entryId")
  .get(getEntryById)
  .patch(updateEntry)
  .delete(deleteEntry);

export default router;
