import numpy as np

cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.npy_float32, ndim=2] _match_best_only(
        np.ndarray[np.npy_float32, ndim=2] iou_mat,
        np.ndarray[np.npy_float32, ndim=1] iou_thrs,
        np.ndarray[np.npy_bool, ndim=1] is_ignore,
        np.ndarray[np.npy_bool, ndim=1] is_crowd
):
    cdef int num_det, num_gt, num_iou_thr
    num_det, num_gt, num_iou_thr = iou_mat.shape[0], iou_mat.shape[1], \
                                   iou_thrs.shape[0]

    cdef np.ndarray[np.npy_int32, ndim=1] best_iou_gt = iou_mat[:,
                                                        ~is_ignore].max(axis=-1)

    cdef np.ndarray[np.npy_int32, ndim=2] matched_gt = np.empty((num_iou_thr,
                                                                 num_det),
                                                                dtype=np.int32)
    cdef np.ndarray[np.npy_bool, ndim=2] gt_matched = np.zeros((num_iou_thr,
                                                                num_gt),
                                                               dtype=np.bool)
    cdef np.npy_float32 iou
    cdef np.npy_int32 matching_gt

    for iou_thr_idx in range(num_iou_thr):
        for det_idx in range(num_det):
            iou = iou_thrs[iou_thr_idx]
            matching_gt = -1
            det_best_iou_gt = best_iou_gt[det_idx]
            for gt_idx in range(num_gt):
                if gt_matched[iou_thr_idx, gt_idx] and (not is_crowd[gt_idx]):
                    continue
                # if dt matched to reg gt, and on ignore gt, stop
                if matching_gt > -1 and (not is_ignore[matching_gt]) and \
                        is_ignore[gt_idx]:
                    continue
                # continue to next gt unless better match made
                if iou_mat[det_idx, gt_idx] < iou:
                    continue
                # match successful
                if not is_ignore[gt_idx]:
                    if det_best_iou_gt == iou_mat[det_idx, gt_idx]:
                        matching_gt = gt_idx
                        break
                else:
                    iou = iou_mat[det_idx, gt_idx]
                    matching_gt = gt_idx
            if matching_gt != -1:
                gt_matched[iou_thr_idx, matching_gt] = True
            matched_gt[iou_thr_idx, det_idx] = matching_gt
    return matched_gt

def match_best_only(iou_mat, iou_thrs, is_ignore, is_crowd):
    return _match_best_only(iou_mat, iou_thrs, is_ignore, is_crowd)
