import numpy as np
from scipy.linalg import sqrtm

import torch


def compute_fid_(model, images1, images2):
    from scipy.linalg import sqrtm
    # compute feature vectors for real and generated images, using model
    # compute feature-wise statistics (means and covariances) for the feature vectors
    # compute distance between statistics, using FID formula
    model.eval()
    with torch.no_grad():
        # need to pass input through the entire model (encoder, sampler, decoder) to get features (latent_vector)
        _ = model(images1)
        features1 = model.latent_vector.cpu().numpy()  # (bs, num_features)
        _ = model(images2)
        features2 = model.latent_vector.cpu().numpy()
    
    means1 = features1.mean(0)    # (bs, num_features) --> (num_features)
    means2 = features2.mean(-1)
    
    # calculate mean and covariance statistics
    sigma1 = np.cov(features1, rowvar=False)
    sigma2 = np.cov(features2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((means1 - means2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def compute_fid(model=None, images1=None, images2=None, eps=1e-6):
    # images1 are reference images from test dataset ('cifar', 'celeba', etc) OR a batch of images in torch format
    # images2 are samples from the model we want to evaluate
    # model can be an actual model object, or a string pointing to a saved model checkpoint - must be full model checkpoint, not a dict
    
    # https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    images1 = images1.to(images2.device)
    model = model.to(images2.device)
    
    # compute feature vectors for real and generated images, using model
    # compute feature-wise statistics (means and covariances) for the feature vectors
    # compute distance between statistics, using FID formula
    model.eval()
    with torch.no_grad():
        # need to pass input through the entire model (encoder, sampler, decoder) to get features (latent_vector)
        _ = model(images1)
        features1 = model.latent_vector.cpu().numpy()  # (bs, num_features)
        _ = model(images2)
        features2 = model.latent_vector.cpu().numpy()
    
    mu1 = features1.mean(0)    # (bs, num_features) --> (num_features)
    mu2 = features2.mean(0)
    
    sigma1 = np.cov(features1, rowvar=False)
    sigma2 = np.cov(features2, rowvar=False)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(f'fid calculation produces singular product, adding {eps} to diagonal of cov estimates')
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def compute_confidence_diversity(classifier, images, N=3, alpha=0.7, debug=False):
    """
    Implements components of the original Inception Score idea:
    - classifier confidence in detected objects in an image
    - diversity of detected objects in a batch of images
    
    Input is logits before softmax has been applied 
    
    0. Apply softmax to logits to produce output vector of class probabilities 
    
    Confidence score computation:
    1. Sort output values
    2. Compute ratio of the sum of top-N values to the total sum, for N=1 and N=5
    3. Weight the combination of the two: confidence score CS = a*S(N=1) + (1-a)*S(N=5), for a in [0, 1]
    
    Diversity score computation:
    1. Record highest predicted class ID for each image in the batch --> vector of class IDs
    2. Count frequency of each class ID  (ideally it should be ~num_images/num_classes)
    3. Apply confidence score algo to frequencies: diversity score DS = CS(freq_count(class_ids))
    
    Alternatively, we could:
        1. compute deviations from class_freq to num_images/num_classes for each class
        2. apply confidence score to this vector, or compute MSE/CE between freq vector and vector of num_images/num_classes
        
    Alternatively, in both cases (confidence and diversity), we can simply compute the std of the vector of
    interest (class probabilities or class ID frequencies)
    """
    
    # images shape: (batch_size, 3, 32, 32)
    device = images.device
    classifier = classifier.to(device)
    classifier.eval()
    with torch.no_grad():
        logits = classifier(images)
        predictions = torch.softmax(logits, dim=1)  # predictions shape: (batch_size, num_classes)
        num_classes = predictions.shape[1]
        top1_ratio = (predictions.max(1)[0] / predictions.sum(1)).mean()
        
        if N > 1:
            sorted_predictions = torch.sort(predictions, dim=1, descending=True)  # returns two tensors: (sorted values (min to max), their orig indices).
            top1_ratio_test = (sorted_predictions[0][:, 0]).mean()  # don't need to divide by total_sum because it's 1 (because of softmax)
            assert torch.allclose(top1_ratio, top1_ratio_test, atol=1e-6)
            topN_ratio = (sorted_predictions[0][:, :N].sum(1)).mean()
            confidence_score = alpha * top1_ratio + (1-alpha) * topN_ratio
        else:
            confidence_score = top1_ratio
        
        # compute std or variance between predictions and means (1/num_classes)
        
        # means = torch.ones_like(predictions) / num_classes
        # mse = ((predictions - means) ** 2).sum(1) / num_classes
        # mse_ref = torch.nn.MSELoss()(predictions, means)
        # variance = predictions.var()
        # # var(unbiased=False) will match the mse, otherwise it will be divided by num_classes-1, not num_classes
        # vars = torch.var(predictions, dim=1, unbiased=False)  

        predicted_class_ids = torch.argmax(predictions, dim=1)  # should be vector of class ids
        class_frequencies = [0] * num_classes
        for class_id in predicted_class_ids:
            class_frequencies[class_id] += 1
        
        labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        
        if debug:
            print(f'\nImages are classified as:')
            for l, f in zip(labels, class_frequencies):
                print(l, f)
            print()
        
        class_frequencies = torch.tensor(class_frequencies, dtype=torch.float, device=device)
        
        ## couple of ways to do it:
        # top1_ratio = class_frequencies.max() / class_frequencies.sum()
        
        # if N > 1:
        #     sorted_class_frequencies = torch.sort(class_frequencies, descending=True)
        #     total_sums = sorted_class_frequencies[0].sum()
        #     top1_ratio_test = (sorted_class_frequencies[0][0] / total_sums)
        #     assert torch.allclose(top1_ratio, top1_ratio_test, atol=1e-6)
        #     topN_ratio = (sorted_class_frequencies[0][:N].sum() / total_sums)
        #     diversity_score1 = 1 - (alpha * top1_ratio + (1-alpha) * topN_ratio)
        # else:
        #     diversity_score1 = 1 - top1_ratio
        
        # construct the batch with worst possible diversity of class IDs:
        min_diversity_batch = [0] * (num_classes - 1) + [predictions.shape[0]]
        min_diversity_batch = torch.tensor(min_diversity_batch, dtype=torch.float, device=device)
        # best diversity batch will have std = 0
        # compare current batch diversity to the worst diversity and invert (low score should indicate low diversity)
        diversity_score2 = 1 - class_frequencies.std() / min_diversity_batch.std()
        
    return 100*confidence_score, 100*diversity_score2


def compute_inception_score(classifier, images, eps=1e-7):
    """
    IS = exp(E[KL(confidence, diversity)]), where 
    confidence is the entropy of output distribution for a single image, and 
    diversity is the entropy of the prediction distribution in a batch of images
    entropy H(x) = -Sum(p(x)*log(p(x)) for all possible outcomes x of random variable X (for continuous valued X use integral)
    for entropy in bits, use log with base 2. For example, if p(x) = 0.5, H(x) = - 2 * 0.5 * log2(2^-1) = -1 * -1 * 1 = 1 bit
    
    entropy of confidence should be low, entropy of diversity should be high
    
    IS = exp(H(diversity) - H(confidence))  
    
    IS = exp[(p_d*log(p_d)).sum().mean() - (p_c*log(p_c)).sum().mean()], where p_d is the distribution of highest outputs from each image in a batch, and
    p_c is the distribution of outputs for a single image
    
    in the official implementation, p_d is computed as an average of all p_c values in the batch (instead of class frequencies)
    
    inputs: 
    classifier: should be trained to classify the same image objects as what we are trying to generate
    images: generated samples we want to evaluate
    """
    
    device = images.device
    classifier = classifier.to(device)
    classifier.eval()
    with torch.no_grad():
        logits = classifier(images)
        predictions = torch.softmax(logits, dim=1)  # predictions shape: (batch_size, num_classes)
        
    # my initial attempt to compute marginal distribution:
    # num_classes = predictions.shape[1]
    # predicted_class_ids = torch.argmax(predictions, dim=1)  # should be vector of class ids
    
    # class_frequencies = [0] * num_classes
    # for class_id in predicted_class_ids:
    #     class_frequencies[class_id] += 1

    # class_frequencies = torch.tensor(class_frequencies, dtype=torch.float, device=device)
    # norm_class_frequencies = class_frequencies/class_frequencies.max()
    # class_frequencies = torch.softmax(norm_class_frequencies, dim=0)
      
    # unit tests:
    #values = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]           # IS should be 3
    #values = [[0.33, 0.33, 0.33], [0.33, 0.33, 0.33], [0.33, 0.33, 0.33]]  # IS should be 1
    #predictions = torch.tensor(values, dtype=torch.float, device=device)
    
    # the below is a more intuitive way to compute IS:
    confidence_entropy = -(predictions * torch.log(predictions+eps)).sum(1).mean()
    class_frequencies = predictions.mean(0)  # needs many images to approximate the marginal distribution
    diversity_entropy = -(class_frequencies * torch.log(class_frequencies+eps)).sum()
    inception_score = torch.exp(diversity_entropy - confidence_entropy)
    
    # the paper https://arxiv.org/pdf/1606.03498.pdf uses this formula: 
    # inception_score = exp(E(KL(confidence, diversity)))
    # KL(confidence || diversity) = Sum(confidence * log(confidence/diversity)) 
    
    # class_frequencies = class_frequencies.unsqueeze(0)  # --> (1, num_classes)
    # kl_div = predictions * (torch.log(predictions+eps) - torch.log(class_frequencies+eps))
    # kl_div = kl_div.sum(1).mean()
    # inception_score = torch.exp(kl_div)
    # should be equivalent to the above:
    # (predictions * torch.log(predictions+eps)).sum(1).mean() - (predictions.mean(0) * torch.log(predictions.mean(0)+eps)).sum()
    
    return inception_score