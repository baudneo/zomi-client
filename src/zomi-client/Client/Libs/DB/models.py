"""
Models representing ZoneMinder DataBase Schema.

This is a WIP and is not complete. May need to add support for different versions of ZM.

"""

from pydantic import BaseModel, Field

class ZMDBModel(BaseModel):
    """
    Model representing Top level ZoneMinder DataBase Schema.

    This is a WIP and is not complete. May need to add support for different versions of ZM.
    """
    Config: ConfigModel = Field(..., alias="Config")
    ControlPresets: ControlPresetsModel = Field(..., alias="ControlPresets")
    Controls: ControlsModel = Field(..., alias="Controls")
    Devices: DevicesModel = Field(..., alias="Devices")
    Event_Data: Event_DataModel = Field(..., alias="Event_Data")
    Event_Summaries: Event_SummariesModel = Field(..., alias="Event_Summaries")
    Events: EventsModel = Field(..., alias="Events")
    EventsArchived: EventsArchivedModel = Field(..., alias="EventsArchived")
    Events_Day: Events_DayModel = Field(..., alias="Events_Day")
    Events_Hour: Events_HourModel = Field(..., alias="Events_Hour")
    Events_Month: Events_MonthModel = Field(..., alias="Events_Month")
    Events_Week: Events_WeekModel = Field(..., alias="Events_Week")
    Filters: FiltersModel = Field(..., alias="Filters")
    Frames: FramesModel = Field(..., alias="Frames")
    Groups: GroupsModel = Field(..., alias="Groups")
    Group_Monitors: Group_MonitorsModel = Field(..., alias="Group_Monitors")
    Group_Permissions: Group_PermissionsModel = Field(..., alias="Group_Permissions")
    Logs: LogsModel = Field(..., alias="Logs")
    Manufacturers: ManufacturersModel = Field(..., alias="Manufacturers")
    Maps: MapsModel = Field(..., alias="Maps")
    Models: ModelsModel = Field(..., alias="Models")
    Monitor_Status: Monitor_StatusModel = Field(..., alias="Monitor_Status")
    MonitorPresets: MonitorPresetsModel = Field(..., alias="MonitorPresets")
    Monitors: MonitorsModel = Field(..., alias="Monitors")
    Monitors_Permissions: Monitors_PermissionsModel = Field(..., alias="Monitors_Permissions")
    MontageLayouts: MontageLayoutsModel = Field(..., alias="MontageLayouts")
    Reports: ReportsModel = Field(..., alias="Reports")
    Server_Stats: Server_StatsModel = Field(..., alias="Server_Stats")
    Servers: ServersModel = Field(..., alias="Servers")
    Sessions: SessionsModel = Field(..., alias="Sessions")
    Snapshot_Events: Snapshot_EventsModel = Field(..., alias="Snapshot_Events")
    Snapshots: SnapshotsModel = Field(..., alias="Snapshots")
    States: StatesModel = Field(..., alias="States")
    Stats: StatsModel = Field(..., alias="Stats")
    Storage: StorageModel = Field(..., alias="Storage")
    TriggersX10: TriggersX10Model = Field(..., alias="TriggersX10")
    User_Preferences: User_PreferencesModel = Field(..., alias="User_Preferences")
    Users: UsersModel = Field(..., alias="Users")
    ZonePresets: ZonePresetsModel = Field(..., alias="ZonePresets")
    Zones: ZonesModel = Field(..., alias="Zones")