/* GameCube / Wii: load into cached MEM1 where DOLs usually start. */
ENTRY(_start)
SECTIONS
{
    . = 0x80003100;
    .text : { KEEP(*(.text._start)) *(.text .text.*) }
    .rodata : { *(.rodata .rodata.*) }
    .data : { *(.data .data.* .sdata .sdata.*) }
    . = ALIGN(4);
    .bss (NOLOAD) : { __bss_start = .; *(.bss .bss.* .sbss .sbss.*) . = ALIGN(4); __bss_end = .; }
    . = ALIGN(16);
    . += 0x10000;
    __stack_top = .;
}
