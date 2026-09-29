pic_text2en_us = {
    "Risk Level": "Risk Level",
    "Score": "Score",
    "Reason": "Reason",
    "Bird's-eye view": "Bird's-eye view",
    "AUDIO WARNING: Pedestrian crossing risk detected!": "AUDIO WARNING: Pedestrian crossing risk detected!",
    "Experimental real-time webcam mode": "Experimental real-time webcam mode",
    "ROI-based risk scoring requires camera-specific ROI calibration.": "ROI-based risk scoring requires camera-specific ROI calibration.",
    "Crosswalk Risk Warning - Webcam Demo": "Crosswalk Risk Warning - Webcam Demo",
}

risk_level2en_us = {
    "DANGER": "DANGER",
    "HIGH": "HIGH",
    "MEDIUM": "MEDIUM",
    "LOW": "LOW"
}

# Keys are the English reason templates emitted by
# CrosswalkRiskPipeline.analyze_risk, so the values are the templates themselves.
reason2en_us = {
    "No pedestrian-related risk detected": "No pedestrian-related risk detected",
    "No pedestrian in crosswalk or waiting zone": "No pedestrian in crosswalk or waiting zone",
    "Pedestrian waiting near unsignalized crosswalk": "Pedestrian waiting near unsignalized crosswalk",
    "Pedestrian inside unsignalized crosswalk": "Pedestrian inside unsignalized crosswalk",
    "Pedestrian in {crosswalk} and vehicle critically close":
        "Pedestrian in {crosswalk} and vehicle critically close",
    "Pedestrian in {crosswalk} and short TTC-like risk detected":
        "Pedestrian in {crosswalk} and short TTC-like risk detected",
    "Pedestrian in {crosswalk}, vehicle approaching, and distance decreasing":
        "Pedestrian in {crosswalk}, vehicle approaching, and distance decreasing",
    "Pedestrian in {crosswalk} and high surrogate risk score detected":
        "Pedestrian in {crosswalk} and high surrogate risk score detected",
    "Pedestrian in {crosswalk} and vehicle close":
        "Pedestrian in {crosswalk} and vehicle close",
    "Pedestrian in {crosswalk} and vehicle in approach zone":
        "Pedestrian in {crosswalk} and vehicle in approach zone",
    "Pedestrian in {crosswalk} and elevated surrogate risk score":
        "Pedestrian in {crosswalk} and elevated surrogate risk score",
    "Pedestrian in {crosswalk} with moderate surrogate risk score":
        "Pedestrian in {crosswalk} with moderate surrogate risk score",
}

# Display names only: the CSV log keeps the raw ROI identifier (main_crosswalk).
crosswalk_name2en_us = {
    "main_crosswalk": "main crosswalk",
    "secondary_crosswalk": "secondary crosswalk",
    "crosswalk": "crosswalk",
}

roi_label2en_us = {
    "main crosswalk": "main crosswalk",
    "main approach": "main approach",
    "waiting zone": "waiting zone",
    "secondary crosswalk": "secondary crosswalk",
    "secondary approach": "secondary approach",
    "main": "main",
    "secondary": "secondary",
    "main app": "main app",
    "sec app": "sec app",
}

class_name2en_us = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
}

chart_text2en_us = {
    "Image-space Surrogate Risk Score Timeline": "Image-space Surrogate Risk Score Timeline",
    "Time (seconds)": "Time (seconds)",
    "Risk score (0–100)": "Risk score (0–100)",
    "MEDIUM threshold": "MEDIUM threshold",
    "HIGH threshold": "HIGH threshold",
    "DANGER threshold": "DANGER threshold",
    "HIGH samples": "HIGH samples",
    "DANGER samples": "DANGER samples",
}

LOCALE = {
    "suffix": "en_us",
    "matplotlib_font": ["DejaVu Sans"],
    "pic_text": pic_text2en_us,
    "risk_level": risk_level2en_us,
    "reason": reason2en_us,
    "crosswalk_name": crosswalk_name2en_us,
    "roi_label": roi_label2en_us,
    "class_name": class_name2en_us,
    "chart_text": chart_text2en_us,
}
