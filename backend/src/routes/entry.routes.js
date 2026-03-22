import { Router } from "express";
import {
  createEntry,
  getMyEntries,
  getEntryById,
  updateEntry,
  deleteEntry,
} from "../controllers/entry.controllers.js";
import { verifyJWT } from "../middlewares/auth.middlewares.js";

const router = Router();

// ❌ Abhi ke liye verifyJWT hata do
router.use(verifyJWT);

// Test log
router.post("/", (req, res, next) => {
  console.log("ENTRY ROUTE HIT");
  next();
}, createEntry);

router.get("/", getMyEntries);

router.get("/:entryId", getEntryById);
router.patch("/:entryId", updateEntry);
router.delete("/:entryId", deleteEntry);

export default router;