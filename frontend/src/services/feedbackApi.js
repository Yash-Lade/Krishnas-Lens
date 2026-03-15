import axiosInstance from "./axiosinstance";

export const createFeedbackApi = (payload) =>
  axiosInstance.post("/feedback", payload);

export const getMyFeedbackApi = () =>
  axiosInstance.get("/feedback/my");

export const deleteFeedbackApi = (feedbackId) =>
  axiosInstance.delete(`/feedback/${feedbackId}`);
