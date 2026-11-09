from .base import BaseSource
from .file_loader import FileBaseSource


def get_source_class(config):
    source_type = config['source_data_type']
    if source_type == 'Amazon S3':
        from .s3_loader import S3Source
        return S3Source(config)
    if source_type == 'Box':
        from .box_loader import BoxSource
        return BoxSource(config)
    elif source_type == 'Local File':
        from .local_file import LocalFileSource
        return LocalFileSource(config)
    elif source_type == 'database':
        from .database_loader import DatabaseSource
        return DatabaseSource(config)
    elif source_type == 'Dropbox':
        from .dropbox_loader import DropboxSource
        return DropboxSource(config)
    elif source_type == 'stripe':
        from .stripe_loader import StripeSource
        return StripeSource(config)
    elif source_type == 'zendesk':
        from .zendesk_loader import ZendeskSource
        return ZendeskSource(config)
    elif source_type == 'Google Drive':
        from .google_drive import GoogleDriveSource
        return GoogleDriveSource(config)
    elif source_type == 'Google Cloud Storage':
        from .google_cloud_storage import GoogleCloudStorageSource
        return GoogleCloudStorageSource(config)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")
