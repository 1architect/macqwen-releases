// Batched positioned reads for one decode layer (research, off by default).
//
// Python hands over a table of reads and releases the GIL for the whole call;
// libdispatch runs them on a user-interactive concurrent queue, `width` lanes
// pulling the next read from a shared counter. Each read loops until its row
// is complete. Returns the number of reads that failed.
#include <dispatch/dispatch.h>
#include <errno.h>
#include <stdatomic.h>
#include <stdint.h>
#include <unistd.h>

typedef struct {
    int32_t fd;
    int32_t pad;
    int64_t offset;
    int64_t length;
    uint64_t dst;
} flashnext_read;

int flashnext_batch_pread(const flashnext_read *reads, int32_t count, int32_t width) {
    if (count <= 0) {
        return 0;
    }
    if (width < 1) {
        width = 1;
    }
    if (width > count) {
        width = count;
    }
    __block atomic_int next = 0;
    __block atomic_int errors = 0;
    dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
    dispatch_apply((size_t)width, queue, ^(size_t lane) {
        (void)lane;
        for (;;) {
            int index = atomic_fetch_add(&next, 1);
            if (index >= count) {
                break;
            }
            const flashnext_read *read = &reads[index];
            char *dst = (char *)(uintptr_t)read->dst;
            int64_t done = 0;
            while (done < read->length) {
                ssize_t got = pread(read->fd, dst + done, (size_t)(read->length - done),
                                    (off_t)(read->offset + done));
                if (got > 0) {
                    done += got;
                } else if (got < 0 && errno == EINTR) {
                    continue;
                } else {
                    atomic_fetch_add(&errors, 1);
                    break;
                }
            }
        }
    });
    return atomic_load(&errors);
}
