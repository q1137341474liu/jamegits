.section .text
.globl _start
_start:
    lui x1, 0x1eceb
    nop             # nops in between to prevent hazard
    nop
    nop
    nop
    nop
    lui x2,2
    nop             # nops in between to prevent hazard
    nop
    nop
    nop
    nop
    sw x2,0(x1)
    nop             # nops in between to prevent hazard
    nop
    nop
    nop
    nop
    #lw x3,0(x1)

    slti x0, x0, -256 # this is the magic instruction to end the simulation
