import torch

from .builder import ANCHOR_GENERATORS
from .anchor_generator import YOLOAnchorGenerator


@ANCHOR_GENERATORS.register_module()
class YOLOV4AnchorGenerator(YOLOAnchorGenerator):
    """Anchor generator for YOLOV4.
    """

    def responsible_indices(self, featmap_sizes, gt_bboxes_list, neighbor=3, shape_match_thres=4., device='cuda'):
        """Generate responsible anchor flags of grid cells in multiple scales.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in multiple
                feature levels.
            gt_bboxes_list (list(Tensor)): List of Ground truth boxes, each with shape (n, 4).
            neighbor (int): assign gt to neighbor grid cell. Possible values:
                0: assign prediction responsibility to the only one grid cell where the center of the gt bbox locates
                2: additionally assign prediction responsibility to 2 nearest neighbor grid cells, like what yolo v5 do
                3: additionally assign prediction responsibility to all 3 neighbor grid cells
            shape_match_thres (float): shape matching threshold between base_anchors and gt-bboxes
                matched gt-bboxes and base_anchors shall meet the following requirements:
                    1.0 / shape_match_thres < (height(gt-bboxes) / height(base_anchors)) < shape_match_thres
                    1.0 / shape_match_thres < (width(gt-bboxes) / width(base_anchors)) < shape_match_thres
            device (str, optional): Device where the flags will be put on.
                Defaults to 'cuda'.

        Return:
            list(tuple(torch.Tensor)): responsible indices
        """
        # Build targets for compute_loss(), input targets(x,y,w,h)
        img_id = []

        for ind, gt_bboxes in enumerate(gt_bboxes_list):
            num_gt = gt_bboxes.shape[0]
            img_id.append(gt_bboxes.new_full((num_gt,), ind, dtype=torch.long))

        gt_bboxes = torch.cat(gt_bboxes_list, dim=0)
        img_id = torch.cat(img_id, dim=0)

        indices = []

        if gt_bboxes.shape[0] == 0:
            for _ in range(self.num_levels):
                indices.append((torch.tensor([], device=device, dtype=torch.long),
                                torch.tensor([], device=device, dtype=torch.long),
                                torch.tensor([], device=device, dtype=torch.long)))
            return indices

        gt_xy = (0.5 * (gt_bboxes[:, 2:4] + gt_bboxes[:, :2])).to(device)
        gt_wh = (gt_bboxes[:, 2:4] - gt_bboxes[:, :2]).to(device)

        neighbor_offset = gt_xy.new_tensor([[0, 0],  # current grid
                                            [-1, 0],  # left neighbor grid
                                            [0, -1],  # upper neighbor grid
                                            [1, 0],  # right neighbor grid
                                            [0, 1],  # lower neighbor grid
                                            [-1, -1],  # upper-left neighbor grid
                                            [1, -1],  # upper-right neighbor grid
                                            [1, 1],  # lower-right neighbor grid
                                            [-1, 1]])  # lower-left neighbor grid

        for i in range(self.num_levels):
            feat_h, feat_w = featmap_sizes[i]
            strides = self.strides[i]
            num_base_anchors = self.num_base_anchors[i]

            base_anchors = self.base_anchors[i].to(device)
            base_anchor_wh = base_anchors[:, 2:] - base_anchors[:, :2]

            # perform shape matching between anchors and gt-boxes
            # the shape of result tensor shape_match: (num_anchors, num_gt)
            shape_deviation = gt_wh[None, :, :] / base_anchor_wh[:, None, :]  # wh ratio
            shape_deviation = torch.max(shape_deviation, 1. / shape_deviation).max(dim=2).values
            shape_match = shape_deviation < shape_match_thres
            base_anchor_ind, gt_ind = shape_match.nonzero(as_tuple=True)

            # Offsets
            feat_size = gt_xy.new_tensor([[feat_w, feat_h]])
            strides = gt_xy.new_tensor([strides])

            xy_grid = gt_xy[gt_ind] / strides  # grid xy
            xy_grid_inv = feat_size - xy_grid  # inverse just used for fast calculation of neighbor cell validity
            if neighbor == 0:
                pred_x, pred_y = xy_grid.long().T
                anchor_ind = (pred_y * feat_w + pred_x) * num_base_anchors + base_anchor_ind
            else:
                x_left_ok, y_up_ok = ((xy_grid % 1. < 0.5) & (xy_grid > 1.)).T
                x_right_ok, y_down_ok = ((xy_grid_inv % 1. < 0.5) & (xy_grid_inv > 1.)).T
                if neighbor == 1:
                    neighbor_ok = torch.stack((torch.ones_like(x_left_ok),
                                               x_left_ok,
                                               y_up_ok,
                                               x_right_ok,
                                               y_down_ok))
                    if neighbor_ok.numel() > 0:
                        four_direction_distance = torch.cat((xy_grid, xy_grid_inv), dim=-1) % 1.
                        direction_mask = (
                                four_direction_distance == four_direction_distance.min(dim=-1).values[:, None])
                        neighbor_ok[1:] = neighbor_ok[1:] & direction_mask.T
                elif neighbor == 2:
                    neighbor_ok = torch.stack((torch.ones_like(x_left_ok),
                                               x_left_ok,
                                               y_up_ok,
                                               x_right_ok,
                                               y_down_ok))
                elif neighbor == 3:
                    xy_upleft_ok = x_left_ok & y_up_ok
                    xy_upright_ok = x_right_ok & y_up_ok
                    xy_downright_ok = x_right_ok & y_down_ok
                    xy_downleft_ok = x_left_ok & y_down_ok
                    neighbor_ok = torch.stack((torch.ones_like(x_left_ok),
                                               x_left_ok,
                                               y_up_ok,
                                               x_right_ok,
                                               y_down_ok,
                                               xy_upleft_ok,
                                               xy_upright_ok,
                                               xy_downright_ok,
                                               xy_downleft_ok))
                else:
                    raise NotImplementedError
                num_offset = neighbor_ok.shape[0]
                gt_ind = gt_ind.repeat((num_offset, 1))[neighbor_ok]
                base_anchor_ind = base_anchor_ind.repeat((num_offset, 1))[neighbor_ok]
                xy_grid_all = (xy_grid[None, :, :] + neighbor_offset[:num_offset, None, :])[neighbor_ok]
                pred_x, pred_y = xy_grid_all.long().T
                anchor_ind = (pred_y * feat_w + pred_x) * num_base_anchors + base_anchor_ind

            indices.append((img_id[gt_ind], anchor_ind, gt_ind))

        return indices
