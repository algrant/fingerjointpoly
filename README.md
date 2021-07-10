# fingerjointpoly
Generating finger joint polyhedra suitable for laser cutting.

using models borrowed from http://dmccooey.com
or obj files (as they are very simple to parse).

# Issues
Not easy to visualise how final product will look unless you build it.

Not clear if dihedrals > 180 are handled correctly.

Not clear if material thickness goes _out_ of face normals or _in_.

## Notes on running with conda
### setup env
```
conda create -n fingerJointPoly
conda activate fingerJointPoly


# install svgwrite
conda install -c conda-forge svgwrite

# install numpy
conda install -c anaconda numpy

# install pyclipper
conda install -c conda-forge pyclipper

# install bottle for api
conda install bottle
```

### run (for now...)
```
# modify & run
python dmcooeyParser.py

# or 
python objParser.py
```


### run viewer
```
# one term
python api/server.py

# another for webpack/hot module reload
cd fjp-react
yarn start

# load on http://127.0.0.1:3000 so cors is happy
```