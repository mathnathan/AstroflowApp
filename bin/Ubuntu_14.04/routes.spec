# -*- mode: python -*-

block_cipher = None


a = Analysis(['../../routes.py'],
             pathex=['/home/ndc08/code/projects/AstroflowApp/src', '/home/ndc08/Dropbox/code/projects/AstroflowApp/bin'],
             binaries=None,
             datas=None,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['matplotlib', 'PyQt4', 'tikinter', 'skimage',
             'gala', 'sympy', 'sklearn', 'PyQt5', 'astropy', 'bokeh', 
             'llvmlite', 'statsmodels', 'notebook', 'theano', 'spyderlib',
             'lxml', 'config', 'babel', 'tables', 'numba', 'nltk', 
             'pip', 'pygments', 'sphinx'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='routes',
          debug=False,
          strip=False,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='routes')
