# Fluid Sim

This is a fluid simulation run on the GPU using WGPU & Rust. The fluid simulation 
is done by breaking the simulation area into a grid of cells with an amount of gas
with a velocity and at a temperature. Based on this velocity and temperature, cells 
will interact with adjacent cells (including diagonally) accordingly. 

## Controls:
- M1: drag to draw a line of gas
- M2: get the contents of a single cell
- 1-3: set the density of the corresponding color (RGB) gas in newly added gas
- 4, 5: set the velocity of newly added gas (x, y)
- 6: set the temperature of newly added gas
- 7: set the mode of the cells to gas is newly added:
  -  0: Color
      -  0: Color behaves normally, each individual color conserved
      -  1: The hue of the cell remains constant, conserves overall gas amount
      -  2: The colored gas in the cell remains constant, same color and amount
  -  1: Velocity
      -  0: Velocity behaves normally
      -  1: Velocity of the cell remains constant
  -  2: Heat
      -  0: Heat behaves normally,
      -  1: Heat of the cell remains constant
  -  3: Wall
      -  0: The Cell is not a wall
      -  1: The Cell is a wall (reflects all adjacent interactions), the contents remain static 
- 8: set the render mode:
  - 0: amount of gas contained and the color of said gas
  - 1: direction (hue) and magnitude (brightness) of velocity
  - 2: direction (hue) and energy (brightness) of velocity
  - 3: temperature (brightness)
  - 4: heat energy (brightness)
  - 5: amount of gas, shifts between colors (R -> G -> B) as amount increases, brightness constant
- r: computes and renders the result of 60 ticks
- f: starts the simulation loop
- q: calculates and renders the result of a "movement" tick (interactions from velocity)
- e: calculates and renders the result of a "diffusion" tick (interactions from heat)
- d: calculates a diagnostic giving totals for the whole simulation area
- v: sets ticks per frame (60 default)
- w: adds a "window" of gas around the simulation area
- g: applies a frame from a supplied file (ie. bad_apple.mp4) to the screen (heats up/cools down based on brightness of pixels)
- t: toggles if frames are applied during simulation loop
- b: sets a file to apply frames from (ie. bad_apple.mp4)

## Limitations:
- velocity is between -1 & 1 cell per tick in the x & y directions
- heat is between 0 & 1 cell per tick (outward)
- can be laggy (use v to drop tick rate if needed)
- interactions due to heat are very approximate and have incorrect magnitudes
