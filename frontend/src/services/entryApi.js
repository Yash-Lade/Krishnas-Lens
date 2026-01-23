import axiosInstance from "./axiosinstance";

// ✅ Create Entry
export const createEntryApi = async (payload) => {
  const { data } = await axiosInstance.post("/entries", payload);
  return data;
};

// ✅ Get My Entries (History)
export const getMyEntriesApi = async () => {
  const { data } = await axiosInstance.get("/entries");
  return data;
};

// ✅ Get Single Entry by ID
export const getEntryByIdApi = async (entryId) => {
  const { data } = await axiosInstance.get(`/entries/${entryId}`);
  return data;
};

// ✅ Update Entry
export const updateEntryApi = async (entryId, payload) => {
  const { data } = await axiosInstance.patch(`/entries/${entryId}`, payload);
  return data;
};

// ✅ Delete Entry
export const deleteEntryApi = async (entryId) => {
  const { data } = await axiosInstance.delete(`/entries/${entryId}`);
  return data;
};
