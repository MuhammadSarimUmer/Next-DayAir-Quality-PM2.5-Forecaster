# ⚡ Quick Test Checklist (15 Minutes)

**Application Status:** ✅ RUNNING  
**Test This:** http://localhost:8502

---

## 🎯 Priority Tests (Do These First)

### ✅ Test 1: Basic Prediction (2 min)
1. Open http://localhost:8502
2. Verify forecast card shows:
   - [ ] City name
   - [ ] PM2.5 prediction value
   - [ ] AQI number and category
   - [ ] Colored background
   - [ ] Health advice text
   - [ ] Action recommendations list

**Expected:** Should see a forecast with Hazardous (Lahore) or Unhealthy (Karachi) category

---

### ✅ Test 2: City Switching (1 min)
1. Select "lahore" from dropdown
2. Note the PM2.5 and category
3. Select "karachi" from dropdown
4. Verify prediction updates

**Expected:** Lahore should show higher PM2.5 than Karachi

---

### ✅ Test 3: Live vs Offline Mode (2 min)
1. Input source: "Live APIs" → Check "Data source" field
2. Input source: "Saved snapshot only" → Check "Data source" field

**Expected:**
- Live mode: "Data source: Live" + staleness < 24h
- Offline mode: "Data source: Offline" + staleness > 300h + warning

---

### ✅ Test 4: Charts Display (2 min)
1. Scroll to "Recent trend and tomorrow's forecast"
2. Check left chart (line chart)
3. Check right chart (bar chart)

**Expected:**
- [ ] Line chart shows historical PM2.5 trend
- [ ] Bar chart shows "Now" vs "Tomorrow (+24h)"
- [ ] No error messages

---

### ✅ Test 5: Manual AQI Calculator (2 min)
1. Expand "Manual AQI / advice simulator"
2. Move slider to these values:
   - 5 → Should show "Good"
   - 20 → Should show "Moderate"
   - 50 → Should show "Unhealthy for Sensitive Groups"
   - 85 → Should show "Unhealthy"
   - 150 → Should show "Very Unhealthy"
   - 300 → Should show "Hazardous"

**Expected:** Category and advice update instantly for each value

---

### ✅ Test 6: Model Accuracy Panel (2 min)
1. Expand "How accurate is the model?"
2. Check metrics are displayed
3. Scroll through the 3 figures

**Expected:**
- [ ] MAE: ~13.31 µg/m³
- [ ] AQI accuracy: ~71.7%
- [ ] Error scatter plot displays
- [ ] Monthly error chart displays
- [ ] Confusion matrix displays

---

### ✅ Test 7: API Health Check (1 min)
1. Open http://127.0.0.1:8001/health
2. Verify JSON response

**Expected:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "n_features": 47
}
```

---

### ✅ Test 8: API Prediction (2 min)
1. Open http://127.0.0.1:8001/docs
2. Expand "POST /predict"
3. Click "Try it out"
4. Enter: `{"city": "lahore", "offline": false}`
5. Click "Execute"

**Expected:**
- Response Code: 200
- predicted_pm25: positive number
- category: one of 6 EPA categories
- actions: array of recommendations

---

### ✅ Test 9: Backend Switching (1 min)
1. In Streamlit sidebar, Backend: "Local pipeline"
2. Get a prediction
3. Backend: "FastAPI service"
4. Get another prediction

**Expected:**
- Local mode: Shows "Input snapshot" table
- FastAPI mode: Works without errors
- Both modes: Same prediction for same inputs

---

### ✅ Test 10: Refresh Button (1 min)
1. Note current prediction
2. Click "Refresh inputs" button
3. Wait for reload

**Expected:**
- [ ] Cache clears
- [ ] Prediction may update
- [ ] No errors

---

## 🎨 Visual Verification

### AQI Color Coding
Check that forecast card background matches category:

| Category | PM2.5 | Color |
|----------|-------|-------|
| Good | 0-9 | 🟢 Green |
| Moderate | 9-35 | 🟡 Yellow |
| Unhealthy for Sensitive | 35-55 | 🟠 Orange |
| Unhealthy | 55-125 | 🔴 Red |
| Very Unhealthy | 125-225 | 🟣 Purple |
| Hazardous | 225+ | 🟤 Maroon |

---

## 🚨 Error Tests (Optional)

### Test Invalid API Input
1. Go to http://127.0.0.1:8001/docs
2. POST /predict: `{"city": "islamabad", "offline": false}`

**Expected:** Response 422 with validation error

### Test Backend Down
1. Stop FastAPI (Ctrl+C in its terminal)
2. Try to get prediction in Streamlit (FastAPI mode)

**Expected:** Friendly error message, app doesn't crash

---

## 📊 Expected Current Values (November 12, 2026)

Based on live data:

**Lahore:**
- Observed: ~263 µg/m³
- Predicted: ~273 µg/m³
- Category: Hazardous (AQI ~336)
- Staleness: ~2-3 hours

**Karachi:**
- Observed: ~48 µg/m³
- Predicted: ~55 µg/m³
- Category: Unhealthy for Sensitive Groups (AQI ~150)
- Staleness: ~2-3 hours

*(Values will change as new data comes in)*

---

## ✅ Success Criteria

Your test is successful if:

- [x] Streamlit app loads without errors
- [x] Predictions display for both cities
- [x] AQI categories have correct colors
- [x] Health advice is contextual
- [x] Charts render properly
- [x] Manual calculator works
- [x] API responds with 200 status
- [x] Mode switching works
- [x] No Python errors in terminals

---

## 🐛 Common Issues Quick Fix

| Issue | Quick Fix |
|-------|-----------|
| Page won't load | Hard refresh: Ctrl+Shift+R |
| Prediction not updating | Click "Refresh inputs" |
| Chart not showing | Check browser console (F12) |
| API error | Verify http://127.0.0.1:8001/health returns 200 |
| Stale data warning | Normal in offline mode, click refresh in live mode |

---

## 📝 After Testing

If everything works:
- ✅ Mark tests as PASS in `reports/manual_testing_guide.md`
- ✅ Note any observations or unusual behavior
- ✅ Report any bugs using the template in the guide

If something fails:
- ❌ Document the error
- ❌ Check troubleshooting section in manual testing guide
- ❌ Report using bug template

---

## 🎉 You're Ready!

Start testing at: **http://localhost:8502**

For detailed testing: See `reports/manual_testing_guide.md`

---

**Created:** November 12, 2026  
**Quick Test Duration:** 15 minutes  
**Full Test Duration:** 1-2 hours (see manual testing guide)
