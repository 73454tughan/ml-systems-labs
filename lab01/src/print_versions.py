import sys
for pkg in ['numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'seaborn', 'torch', 'torchvision', 'torchinfo', 'thop', 'psutil', 'joblib']:
    try:
        m = __import__(pkg)
        print(f"{pkg}: {getattr(m, '__version__', 'installed')}")
    except ImportError:
        print(f"{pkg}: not installed")
