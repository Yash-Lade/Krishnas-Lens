import mongoose from "mongoose";
import { db_name } from "../constant.js";

const connectDB = async () => {
  try {
    const connectionInstance = await mongoose.connect(
      `${process.env.MONGODB_URI}/${db_name}`
    );

    console.log(
      `MongoDB connected !! DB HOST: ${connectionInstance.connection.host}`
    );

    // ✅ NOTE:
    // Krishna's Lens me "Group" / "Expense" models nahi hain,
    // so syncIndexes/createIndexes wala purana code REMOVE.
  } catch (error) {
    console.log("MONGODB connection Failed", error);
    process.exit(1);
  }
};

export default connectDB;
