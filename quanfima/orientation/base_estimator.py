import os
import numpy as np

class OrientationEstimatorBase:
    def orientation_in_batches(
        self,
        sample_name,
        skel,
        data,
        window_size,
        output_dir,
        out_arr_names=['lat', 'azth'], 
        batch_size=100,
        make_output=True,
        method_name=None,
    ):
        output = dict()
        output_props = dict()

        for arr_name in out_arr_names:
            output[arr_name] = np.zeros_like(data, dtype=np.float32)

        output_props['time'] = 0.
        output_props['sample_name'] = sample_name

        depth, height, width = data.shape
        batches_idxs = np.array_split(np.arange(depth), np.ceil(depth / float(batch_size)))
        
        cls_name = __class__.__name__
        border_gap = window_size
        
        for batch_idxs in batches_idxs:
            batch_len = len(batch_idxs)

            if batch_idxs[0] == 0:            
                arr1, arr2 = None, np.arange(batch_idxs[-1]+1, batch_idxs[-1]+border_gap+1)
                gaped_batch_idxs = np.concatenate((batch_idxs, arr2))
            elif batch_idxs[-1] == (depth - 1):
                arr1, arr2 = np.arange(batch_idxs[0]-border_gap, batch_idxs[0]), None
                gaped_batch_idxs = np.concatenate((arr1, batch_idxs))
            else:
                arr1, arr2 = np.arange(batch_idxs[0]-border_gap, batch_idxs[0]), \
                                np.arange(batch_idxs[-1], batch_idxs[-1]+border_gap+1)        
                gaped_batch_idxs = np.concatenate((arr1, batch_idxs, arr2))

            batched_skel = skel[gaped_batch_idxs]
            batched_data = data[gaped_batch_idxs]

            gaped_out_dict = self.orientation(batched_skel, batched_data, window_size, do_reshape=False)

            output_props['time'] += gaped_out_dict['time']

            for data_name in out_arr_names:
                gaped_arr_values = gaped_out_dict[data_name]

                gaped_arr = np.zeros_like(batched_data, dtype=np.float32)
                gaped_arr[batched_skel.nonzero()] = gaped_arr_values

                if batch_idxs[0] == 0:
                    output[data_name][batch_idxs] = gaped_arr[:batch_len]
                elif batch_idxs[-1] == (depth - 1):
                    output[data_name][batch_idxs] = gaped_arr[border_gap:]
                else:
                    output[data_name][batch_idxs] = gaped_arr[border_gap:border_gap+batch_len]

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if method_name is not None:
            print('Total {} time: {}'.format(method_name, output_props['time']))
        else:
            print('Total time: {}'.format(output_props['time']))

        if make_output:
            opath = os.path.join(output_dir, '{}_w{}_{}_orientation_evaluation.npy').format(
                sample_name,
                window_size,
                cls_name,
            )
            output['props'] = output_props
            np.save(opath, output)

        return output_props
    
    def orientation_in_single_batch(
        self,
        sample_name,
        skel,
        data,
        window_size,
        output_dir,
        out_arr_names=['lat', 'azth'], 
        make_output=True,
        method_name=None,
    ):
        output = dict()
        
        cls_name = __class__.__name__
        orient = self.orientation(skel, data, window_size, do_reshape=False)
        
        output_props = {
            'time': orient['time'],
            'sample_name': sample_name,
        }
        
        for data_name in out_arr_names:
            output_values = orient[data_name]

            result = np.zeros_like(data, dtype=np.float32)
            result[skel.nonzero()] = output_values
            
            output[data_name] = result
            
        if method_name is not None:
            print('Total {} time: {}'.format(method_name, output_props['time']))
        else:
            print('Total time: {}'.format(output_props['time']))

        if make_output:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            opath = os.path.join(output_dir, '{}_w{}_{}_orientation_evaluation.npy').format(
                sample_name,
                window_size,
                cls_name,
            )
            output['props'] = output_props
            np.save(opath, output)

        return output_props

    def orientation(self, skel, data, window_size, n_lat=90, n_azth=180, do_reshape=True):
        raise NotImplementedError('The method should be implemented in child classes.')
