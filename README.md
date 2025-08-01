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

### 4. View the results

After analysis, a new folder named **Results** will be created inside
the *Data* directory:

<pre> ``` 📁 src  ├── 📁 Data  └── 📁 Results  └── ... analysis output files ``` </pre>


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
audio_id ,audio_title, naturalness, emotions, rhythm, labeling_date,               is_deepfake,state  
4        ,3.wav      ,3            ,3        ,2       ,2025-06-13 10:54:32.772599  ,false      ,3

### Contributing

Contributions, feature requests, and bug reports are welcome!

### License

This project is licensed under the MIT License.

### Author

GitHub: @FlakDemaj  
Email: flakron.demaj@outlook.de
