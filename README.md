# LabeledAudioAnalyzer

A tool for analyzing labeled audio files - automated, structured, and easy to use

---

## Features

- Automatically analyze labeled audio files
- Creates a clean and structured `Results` file
- Output ready for further processing and evaluation

---

## Quick Start

### 1. Clone the repository (optional)

```bash
git clone https://github.com/FlakDemaj/LabeledAudioAnalyzer.git
cd LabeledAudioAnalyzer
```

### 2. Install dependencies

To install all the needed dependencies, just use the following command

```bash
pip install -r requirements.txt
```

### 3. Run the analysis

Start the analysis by executing the function `main()` in main.py or with
the following command:

```bash
python main.py
```

After starting, you will be asked whether you want to display the plots.
You can only answer this question with yes or no.
Regardless of your choice, the plots will always be saved.

This prompt is especially relevant if you run the analyzer in a terminal:
After each plot is generated, the program will pause and display it until you close the window.
With many plots, this can quickly become annoying.

**Recommendation:**  
Choose no if you run the analyzer in a terminal.
If you use an IDE or a similar environment, this issue will not occur.

### 5. View the results

After analysis, a new folder named **Results** will be created inside
the *Data* directory:

📁 src                 
├── 📁 Data  
-------└── 📁 Results  
--------------└── ... analysis output files


## General

### Project Structure

📁 LabeledAudioAnalyzer  
 ├── 📁 src                # Modules / Data  
 ├── main.py               # Entry point for analysis  
 ├── requirements.txt      # Python dependencies  
 ├── README.md             # Project documentation  
 ├── LICENSE               # The License to the project 
 └── .gitignore            # Git ignore file  
 
### Requirements
 
 * Python 3.8 or higher
 * OS_ Windows, macOS or Linux

### Sample Data
- audio_id: 4 
- audio_title: 3.wav
- naturalness: 3
- emotions: 3
- rhythm: 2
- labeling_date: 2025-06-13 10:54:32.772599
- is_deepfake: false
- state: 3  

### Contributing

Contributions, feature requests, and bug reports are welcome!

### License

This project is licensed under the MIT License.

### Author

GitHub: @FlakDemaj  
Email: flakron.demaj@outlook.de
