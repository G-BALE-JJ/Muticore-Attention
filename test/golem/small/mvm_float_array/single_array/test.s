	.file	"single_array2.cpp"
	.option nopic
	.text
	.section	.srodata,"a"
	.align	3
	.type	_ZStL19piecewise_construct, @object
	.size	_ZStL19piecewise_construct, 1
_ZStL19piecewise_construct:
	.zero	1
	.section	.rodata
	.align	3
.LC0:
	.string	"Matrix:"
	.align	3
.LC2:
	.string	"%.1f "
	.align	3
.LC3:
	.string	"\nVector:"
	.align	3
.LC5:
	.string	"\n\nOutput Vector:"
	.text
	.align	1
	.globl	main
	.type	main, @function
main:
.LFB861:
	.cfi_startproc
	addi	sp,sp,-80
	.cfi_def_cfa_offset 80
	sd	ra,72(sp)
	sd	s0,64(sp)
	.cfi_offset 1, -8
	.cfi_offset 8, -16
	addi	s0,sp,80
	.cfi_def_cfa 8, 0
	li	a5,6
	sw	a5,-36(s0)
	li	a5,6
	sw	a5,-40(s0)
	lw	a4,-36(s0)
	lw	a5,-40(s0)
	mulw	a5,a4,a5
	sext.w	a5,a5
	li	a4,-9
	srli	a4,a4,3
	bgtu	a5,a4,.L2
	slli	a5,a5,2
	mv	a0,a5
	call	_Znam
	mv	a5,a0
	sd	a5,-48(s0)
	lw	a5,-40(s0)
	li	a4,-9
	srli	a4,a4,3
	bgtu	a5,a4,.L4
	j	.L17
.L2:
	call	__cxa_throw_bad_array_new_length
.L17:
	slli	a5,a5,2
	mv	a0,a5
	call	_Znam
	mv	a5,a0
	sd	a5,-56(s0)
	lw	a5,-36(s0)
	li	a4,-9
	srli	a4,a4,3
	bgtu	a5,a4,.L6
	j	.L18
.L4:
	call	__cxa_throw_bad_array_new_length
.L18:
	slli	a5,a5,2
	mv	a0,a5
	call	_Znam
	mv	a5,a0
	sd	a5,-64(s0)
	lui	a5,%hi(.LC0)
	addi	a0,a5,%lo(.LC0)
	call	puts
	sw	zero,-20(s0)
	j	.L11
.L6:
	call	__cxa_throw_bad_array_new_length
.L11:
	lw	a4,-20(s0)
	lw	a5,-36(s0)
	sext.w	a4,a4
	sext.w	a5,a5
	bge	a4,a5,.L8
	sw	zero,-24(s0)
.L10:
	lw	a4,-24(s0)
	lw	a5,-40(s0)
	sext.w	a4,a4
	sext.w	a5,a5
	bge	a4,a5,.L9
	lw	a4,-20(s0)
	lw	a5,-40(s0)
	mulw	a5,a4,a5
	sext.w	a5,a5
	lw	a4,-24(s0)
	addw	a5,a4,a5
	sext.w	a5,a5
	slli	a5,a5,2
	ld	a4,-48(s0)
	add	a5,a4,a5
	lui	a4,%hi(.LC1)
	flw	fa5,%lo(.LC1)(a4)
	fsw	fa5,0(a5)
	lw	a4,-20(s0)
	lw	a5,-40(s0)
	mulw	a5,a4,a5
	sext.w	a5,a5
	lw	a4,-24(s0)
	addw	a5,a4,a5
	sext.w	a5,a5
	slli	a5,a5,2
	ld	a4,-48(s0)
	add	a5,a4,a5
	flw	fa5,0(a5)
	fcvt.d.s	fa5,fa5
	fmv.x.d	a1,fa5
	lui	a5,%hi(.LC2)
	addi	a0,a5,%lo(.LC2)
	call	printf
	lw	a5,-24(s0)
	addiw	a5,a5,1
	sw	a5,-24(s0)
	j	.L10
.L9:
	li	a0,10
	call	putchar
	lw	a5,-20(s0)
	addiw	a5,a5,1
	sw	a5,-20(s0)
	j	.L11
.L8:
	lui	a5,%hi(.LC3)
	addi	a0,a5,%lo(.LC3)
	call	puts
	sw	zero,-28(s0)
.L13:
	lw	a4,-28(s0)
	lw	a5,-40(s0)
	sext.w	a4,a4
	sext.w	a5,a5
	bge	a4,a5,.L12
	lw	a5,-28(s0)
	slli	a5,a5,2
	ld	a4,-56(s0)
	add	a5,a4,a5
	lui	a4,%hi(.LC4)
	flw	fa5,%lo(.LC4)(a4)
	fsw	fa5,0(a5)
	lw	a5,-28(s0)
	slli	a5,a5,2
	ld	a4,-56(s0)
	add	a5,a4,a5
	flw	fa5,0(a5)
	fcvt.d.s	fa5,fa5
	fmv.x.d	a1,fa5
	lui	a5,%hi(.LC2)
	addi	a0,a5,%lo(.LC2)
	call	printf
	lw	a5,-28(s0)
	addiw	a5,a5,1
	sw	a5,-28(s0)
	j	.L13
.L12:
	sw	zero,-68(s0)
	ld	a5,-48(s0)
	lw	a4,-68(s0)
#APP
# 40 "single_array2.cpp" 1
	mvm.set a5, a5, a4
# 0 "" 2
#NO_APP
	sw	a5,-72(s0)
	ld	a5,-56(s0)
	lw	a4,-68(s0)
#APP
# 47 "single_array2.cpp" 1
	mvm.l a5, a5, a4
# 0 "" 2
#NO_APP
	sw	a5,-72(s0)
	lw	a5,-68(s0)
#APP
# 53 "single_array2.cpp" 1
	mvm a5, a5, x0
# 0 "" 2
#NO_APP
	sw	a5,-72(s0)
	ld	a5,-64(s0)
	lw	a4,-68(s0)
#APP
# 60 "single_array2.cpp" 1
	mvm.s a5, a5, a4
# 0 "" 2
#NO_APP
	sw	a5,-72(s0)
	lui	a5,%hi(.LC5)
	addi	a0,a5,%lo(.LC5)
	call	puts
	sw	zero,-32(s0)
.L15:
	lw	a4,-32(s0)
	lw	a5,-36(s0)
	sext.w	a4,a4
	sext.w	a5,a5
	bge	a4,a5,.L14
	lw	a5,-32(s0)
	slli	a5,a5,2
	ld	a4,-64(s0)
	add	a5,a4,a5
	flw	fa5,0(a5)
	fcvt.d.s	fa5,fa5
	fmv.x.d	a1,fa5
	lui	a5,%hi(.LC2)
	addi	a0,a5,%lo(.LC2)
	call	printf
	lw	a5,-32(s0)
	addiw	a5,a5,1
	sw	a5,-32(s0)
	j	.L15
.L14:
	ld	a0,-48(s0)
	call	free
	ld	a0,-56(s0)
	call	free
	ld	a0,-64(s0)
	call	free
	li	a5,0
	mv	a0,a5
	ld	ra,72(sp)
	.cfi_restore 1
	ld	s0,64(sp)
	.cfi_restore 8
	.cfi_def_cfa 2, 80
	addi	sp,sp,80
	.cfi_def_cfa_offset 0
	jr	ra
	.cfi_endproc
.LFE861:
	.size	main, .-main
	.section	.rodata
	.align	2
.LC1:
	.word	1065353216
	.align	2
.LC4:
	.word	1073741824
	.ident	"GCC: (GNU) 9.4.0"
	.section	.note.GNU-stack,"",@progbits
