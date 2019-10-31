# NaturalLanguageProcessingApplication
The system will read specification and then create scripts:

Demos:

Input:
Precondition: 
1. Set Ubatt = 13.8 V
2. Ignition ON
3. Set Vehicle speed = 0 kph
Test Procedure:
1. Start MM6 measurement
2. Set Vehicle speed to 100 kph.
3. Wait 5s
4. Set Vehicle speed to 0 kph
5. Stop MM6 measurement
6. Read RB and CU memory.

Output:
SC_set_Ubatt

SC_ignition_on

SC_goto_target_speed

SC_EA_trace_start

SC_goto_target_speed

SC_wait_ms

SC_goto_target_speed

SC_EA_trace_stop

DCOM_read_and_eval_fcm_rb_cu


