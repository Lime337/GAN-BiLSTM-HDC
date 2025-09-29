# Create a clock constraint for 100 MHz
create_clock -period 10.000 -name clk -waveform {0.000 5.000} [get_ports clk]

set_switching_activity -toggle_rate 12.500 -type {lut} -static_probability 0.500 -all 
set_switching_activity -toggle_rate 12.500 -type {register} -static_probability 0.500 -all 
set_switching_activity -toggle_rate 12.500 -type {shift_register} -static_probability 0.500 -all 
set_switching_activity -toggle_rate 12.500 -type {lut_ram} -static_probability 0.500 -all 
set_switching_activity -toggle_rate 12.500 -type {bram} -static_probability 0.500 -all 
set_switching_activity -toggle_rate 12.500 -type {dsp} -static_probability 0.500 -all 
set_switching_activity -toggle_rate 12.500 -type {io_output} -static_probability 0.500 -all 
