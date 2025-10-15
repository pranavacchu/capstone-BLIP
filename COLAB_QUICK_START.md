# 🚀 Google Colab Quick Start Guide

## How to Use the Notebook

### Step 1: Open in Google Colab

1. Upload `Video_Search_Colab.ipynb` to your Google Drive
2. Right-click the file → Open with → Google Colaboratory
3. **OR** Go to https://colab.research.google.com/ and upload the file

### Step 2: Enable GPU

🔥 **IMPORTANT**: Enable GPU for faster processing!

1. Click `Runtime` in the menu
2. Select `Change runtime type`
3. Choose `T4 GPU` (or any available GPU)
4. Click `Save`

### Step 3: Run the Cells in Order

Simply run each cell from top to bottom by clicking the ▶️ play button or pressing `Shift + Enter`.

**Cell Execution Order:**
1. ✅ Clone repository & install dependencies (3-5 min)
2. ✅ Configure Pinecone API key
3. ✅ Test connection
4. ✅ Upload your video
5. ✅ Process the video (2-10 min depending on video length)
6. ✅ Search your content!

---

## 📝 What Each Step Does

| Step | Time | Description |
|------|------|-------------|
| **Step 1** | ~3-5 min | Clones GitHub repo, installs all packages |
| **Step 2** | ~5 sec | Sets up your Pinecone API credentials |
| **Step 3** | ~10 sec | Tests connection to Pinecone |
| **Step 4** | Varies | Upload or download video file |
| **Step 5** | 2-10 min | Processes video with AI |
| **Step 6** | <1 sec | Search with natural language |
| **Step 7-11** | <1 sec | Advanced features |

---

## 💡 Tips for Success

### Before You Start
- ✅ Have a video file ready (MP4 works best)
- ✅ Video should be < 500MB for smooth upload
- ✅ Start with a 1-2 minute video for testing

### During Processing
- ⏳ Don't close the browser tab while processing
- 📊 Watch the progress bars
- ⚠️ If you get GPU memory errors, restart runtime

### When Searching
- 🔍 Use descriptive queries ("person with blue jacket")
- 📝 Try different phrasings if no results
- 🎯 Start broad, then get specific

---

## 🎬 Example Workflow

```
1. Upload video: "vacation_2024.mp4"
2. Process it (wait 5 minutes)
3. Search queries:
   - "beach sunset"
   - "person swimming"
   - "kids playing in sand"
4. Get exact timestamps!
```

---

## ⚡ Performance Expectations

| Hardware | Processing Speed | Search Speed |
|----------|-----------------|--------------|
| **T4 GPU** | ~2-3 min/min of video | <1 second |
| **CPU** | ~8-10 min/min of video | 1-2 seconds |

**Example**: A 5-minute video takes ~10-15 minutes to process on T4 GPU

---

## 🐛 Common Issues & Solutions

### "No GPU available"
- Go to `Runtime` → `Change runtime type` → Select GPU

### "ModuleNotFoundError"
- Re-run Step 1 (installation cell)

### "Video upload failed"
- Try a smaller video file
- Convert to MP4 format
- Use Option 2 (URL download) for large files

### "Out of memory"
- Restart runtime: `Runtime` → `Restart runtime`
- Use a shorter video
- Close other browser tabs

### "No results found"
- Try different search terms
- Lower the similarity threshold
- Make sure video was fully processed

---

## 📱 Mobile Usage

Yes, you can use this on mobile!
- Use Chrome or Safari browser
- Upload smaller videos (<100MB)
- Processing will be slower on mobile

---

## 💾 Saving Your Work

**Important**: Colab automatically clears after inactivity!

To save your database:
- Your data is stored in Pinecone (persistent)
- Videos and frames are temporary
- Search anytime from any device with the notebook

---

## 🎯 Quick Commands

In any cell, you can run:

```python
# Quick search
engine.search("your query", top_k=5)

# View database stats
engine.get_index_stats()

# Process another video
engine.process_video("video2.mp4")
```

---

## 📊 Understanding Results

When you search, you'll see:
- ⏱️ **Timestamp**: Exact time in video (MM:SS.ms)
- 📝 **Caption**: What AI saw in that frame
- 📊 **Confidence**: How well it matches (60%+ is good)
- 🎥 **Video**: Which video it's from

**Score Guide:**
- 🟢 70%+: Excellent match
- 🟡 50-70%: Good match
- 🟠 40-50%: Possible match

---

## 🔗 Direct Colab Link

Once uploaded to GitHub, you can create a direct link:
```
https://colab.research.google.com/github/pranavacchu/capstone-BLIP/blob/main/Video_Search_Colab.ipynb
```

---

## 📞 Need Help?

If you encounter issues:
1. Check the error message in the cell output
2. Re-run the problematic cell
3. Restart runtime if needed
4. Check GitHub repo for updates

---

**Happy Searching! 🎉**