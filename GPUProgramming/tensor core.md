mma字段解释

三代 tensor core 的pingjingyuduice

sm80
全在寄存器
fragment ldmatrix swizzle

sm90
寄存器带宽不够 且tc计算时cuda core空闲
descriptor 异步wgamma TMA mbarrier

sm100
累加器占用大量寄存器
发射指令不需要线程参与
TMEM 单线程发射 2-CTA


