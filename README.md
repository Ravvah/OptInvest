# OptInvest 

## Description

Optimize your investment portfolio with OptInvest.



## Installation

1. clone the project from GitHub

2. install uv

```bash
pip install uv
```



3. create venv and activate it

```bash
uv venv
```

```bash
source .venv/bin/activate
```


4. install dependencies in the venv

```bash
uv add -r requirements.txt
```

NB : you can also use `uv pip install -r requirements.txt` if `uv add` provokes an error


5. run the api back-end

```bash
python -m app.api.main
```

6. run the web front-end

In another terminal

```bash	
streamlit run app/client/main.py
```

7. access to the app in the browser : `http://localhost:8501`














