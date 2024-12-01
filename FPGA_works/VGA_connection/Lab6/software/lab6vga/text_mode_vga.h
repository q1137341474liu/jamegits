/*
 *  text_mode_vga.h
 *	Minimal driver for text mode VGA support, ECE 385 Summer 2021 Lab 6
 *  You may wish to extend this driver for your final project/extra credit project
 * 
 *  Created on: Jul 17, 2021
 *      Author: zuofu
 */

#ifndef TEXT_MODE_VGA_H_
#define TEXT_MODE_VGA_H_

#define COLUMNS 80
#define ROWS 30

//define some colors
#define WHITE 		0xFFF
#define BRIGHT_RED 	0xF00
#define DIM_RED    	0x700
#define BRIGHT_GRN	0x0F0
#define DIM_GRN		0x070
#define BRIGHT_BLU  0x00F
#define DIM_BLU		0x007
#define GRAY		0x777
#define BLACK		0x000


struct TEXT_VGA_STRUCT {
	unsigned short VRAM [ROWS*COLUMNS];
	unsigned char CTRL;
};

//you may have to change this line depending on your platform designer
static volatile struct TEXT_VGA_STRUCT* vga_ctrl = 0x1000;

void textVGASetColor(int background, int foreground);
void textVGAClr();
void textVGATest();

#endif /* TEXT_MODE_VGA_H_ */
