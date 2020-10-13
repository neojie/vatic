"""
VATIC - Vasp : smAll ToolkIt Collection
======
JDlib
- trajxdt -> view the XDATCAR
Below is the template from PythEoS 
"""
"""
PythEOS
=======
"""
#### Frequency used, make "make vatic.XX" avaialable ####
# this make vatic.XDATCAR_toolkit direclty usable, but other .py file in others not usable
from .others import XDATCAR_toolkit
from .trajectory import trajxdt_v3
from .tcdac import tcdac_v2

#### import others folder 
from . import others
from .others import *

#### import interpy folder 
from . import interpy
from .interpy import *
from .TI import ti

#### import IR folder 
from . import IR
from .IR import *

#### import plots folder
from . import plots
from .plots import *

