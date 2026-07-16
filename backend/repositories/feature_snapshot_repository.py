from configs.logging import app_logger

from backend.database.models.feature_snapshot import (
    FeatureSnapshot,
)


class FeatureSnapshotRepository:

    def __init__(self, db):

        self.db = db

    def create(self, snapshot: FeatureSnapshot):

        try:

            self.db.add(snapshot)

            self.db.commit()

            self.db.refresh(snapshot)

            return snapshot

        except Exception:

            self.db.rollback()

            app_logger.exception(
                "FeatureSnapshotRepository.create failed"
            )

            raise

    def get_last(self, limit=100):

        return (

            self.db.query(FeatureSnapshot)

            .order_by(
                FeatureSnapshot.id.desc()
            )

            .limit(limit)

            .all()

        )

    def get_by_symbol(

        self,

        symbol,

        limit=100,

    ):

        return (

            self.db.query(FeatureSnapshot)

            .filter(

                FeatureSnapshot.symbol == symbol

            )

            .order_by(

                FeatureSnapshot.id.desc()

            )

            .limit(limit)

            .all()

        )

    def count(self):

        return (

            self.db.query(

                FeatureSnapshot

            )

            .count()

        )