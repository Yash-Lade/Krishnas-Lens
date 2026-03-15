import dotenv from "dotenv";
dotenv.config();

import connectDB from "./db/db.js";
import { app } from "./app.js";

const PORT = process.env.PORT || 5000;

connectDB()
  .then(() => {
    console.log("MongoDB connected ✅");

    app.listen(PORT, () => {
      console.log(`Server running on port : ${PORT}`);
    });
  })
  .catch((err) => {
    console.log("MongoDB connection failed ❌", err?.message || err);
    process.exit(1);
  });
