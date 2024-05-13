
def find_semihard_exmaple(pos_col, pos_row, neg):
    """ find semi-hard examples
    """
    neg_mask_col = tf.cast(tf.greater(neg, pos_col), tf.float32)
    neg_mask_row = tf.cast(tf.greater(neg, pos_row), tf.float32)
    neg_select_col = tf.reduce_min(tf.multiply(neg_mask_col, neg) +
                                   tf.multiply(1.0 - neg_mask_col, tf.reduce_max(neg)),
                                   axis=0)
    neg_select_row = tf.reduce_min(tf.multiply(neg_mask_row, neg) +
                                   tf.multiply(1.0 - neg_mask_row, tf.reduce_max(neg)),
                                   axis=1)

    semihard_neg_select = tf.concat([neg_select_col, neg_select_row], axis=0)

    return semihard_neg_select


def triplet_loss_compute_semihard(feature1, feature2, labels, margin=1.0):
    """ triplet loss with semi-hard negative pairs
    """
    batch_size = labels.get_shape().as_list()[0]
    labels = tf.cast(tf.reshape(labels, [batch_size, 1]), tf.float32)

    feature1 = tf.nn.l2_normalize(tf.reshape(feature1, [batch_size, -1]), dim=-1)
    feature2 = tf.nn.l2_normalize(tf.reshape(feature2, [batch_size, -1]), dim=-1)

    cross_feaD = 1.0 - tf.matmul(feature1, tf.transpose(feature2))  # cosine distance

    labelD = pairwise_distance(labels, labels)
    label_mask = tf.cast(tf.greater(labelD, 0.5), tf.float32)  # 0-similar   1-dissimilar

    # num_match = batch_size*batch_size-tf.reduce_sum(tf.reduce_sum(label_mask,0))


    cross_feaD_pos = tf.multiply(1.0 - label_mask, cross_feaD)

    cross_feaD_neg = tf.multiply(label_mask, cross_feaD)


    # haha = tf.concat([cross_feaD_pos,cross_feaD_neg],1)


    cross_pos_col = tf.reduce_max(cross_feaD_pos, axis=0, keep_dims=True)
    cross_pos_row = tf.reduce_max(cross_feaD_pos, axis=1, keep_dims=True)

    semihard_negD_select = find_semihard_exmaple(cross_pos_col, cross_pos_row, cross_feaD_neg)

    cross_posD_select = tf.concat([tf.squeeze(cross_pos_col), tf.squeeze(cross_pos_row)], axis=0)

    margin = FLAGS.margin   # + hist_distance_compute(cross_posD_select, semihard_negD_select)

    pos_select_dist = tf.reduce_mean(cross_posD_select)
    neg_select_dist = tf.reduce_mean(semihard_negD_select)

    loss = tf.reduce_mean(tf.maximum(margin + cross_posD_select - semihard_negD_select, 0.0))

    return loss, pos_select_dist, neg_select_dist, margin

